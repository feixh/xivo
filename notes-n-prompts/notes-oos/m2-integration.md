# M2 — wiring the OOS update into the filter

Scope: `src/estimator.{h,cpp}`, `src/update.cpp`, `src/manager.cpp`,
`cfg/oos.json`. After this milestone `use_OOS: true` runs end to end instead of
aborting with `LOG(FATAL) << "MSCKF not implemented"`.

## Where the measurement is computed, and why there

The natural place to build an OOS measurement is `ProcessTracks`, the moment the
tracker drops the feature. That is wrong here. Between `ProcessTracks` and the
update, `UpdateStep` still runs `SelectAndAddNewFeatures`, `OutlierRejection`
and — the killer — `DiscardAffectedGroups`, which calls `RemoveGroupFromState`
and thereby *frees group state slots and zeroes the corresponding rows and
columns of `P_`*. An OOS Jacobian is indexed by `g->sind()`, so a Jacobian built
before that point can end up pointing at a slot that now belongs to a different
group, or to no group at all.

So the flow is split:

* `ProcessTracks` — a dropped feature that was never in the state is *kept*
  alive in the graph (with its observations) and pushed onto `oos_features_`,
  instead of being removed and destroyed. Its status becomes
  `REJECTED_BY_TRACKER`. That status matters: `Criteria::Candidate` accepts only
  `READY`/`INITIALIZING`, so the feature cannot be promoted into the state
  between here and the update, which would double-count it.
  (`FeatureStatus::DROPPED`, which the deleted MSCKF code used, no longer exists
  in the enum.)
* `ComputeOOSMeasurements()` — called from `UpdateStep` *after* all group
  management, immediately before `FilterUpdate`. Per candidate: select in-state
  views, re-triangulate (`RefineOOSDepth`), build and marginalize the Jacobian
  (`ComputeOOSJacobian`), Mahalanobis-gate. Survivors land in `oos_used_`.
* `CleanupOOSFeatures()` — called after the update; removes the candidates from
  the graph and destroys them.

Two lifetime hazards, both guarded:

* `DiscardAffectedGroups`/`DiscardFeatures` can destroy an OOS candidate in
  between (its reference group went away and no new owner was found). Both
  `ComputeOOSMeasurements` and `CleanupOOSFeatures` therefore test
  `graph.HasFeature(f)` first.
* `MemoryManager::DeactivateItem` clears only `slots_active_`, while
  `DestroyItem` clears active *and* initialized and decrements both counters.
  Calling `Destroy` on an already-`Deactivate`d feature corrupts the pool
  accounting, so `CleanupOOSFeatures` destroys only what is still in the graph.

`UpdateStep` also clears `oos_features_`/`oos_used_` at the top, so a step that
bails out early cannot leak stale pointers into the next one.

## Stacking

`FilterUpdate(int oos_rows)` sizes `H_`/`inn_`/`diagR_` to
`2*|in_current_ekf_update_| + oos_rows` and appends the OOS blocks below the
in-state ones. `f->Ho()` already has `err_.size() == kFullSize` columns, so the
block copy is a straight assignment. `diagR_` gets `Roos_ = oos_meas_std^2`
(3.5^2) on those rows — valid *only* because M1 marginalizes with an orthonormal
nullspace basis, so `A' R A = sigma^2 I`.

The update now fires when there are in-state measurements **or** OOS rows;
previously an all-OOS step would have been skipped.

## Gating

`OOSGating` computes `S = Ho P Ho' + Roos I`, `d = r' S^-1 r`, and rejects when
`d > MH_thresh * n` with `n = 2n_obs - 3` the row count. Normalising per degree
of freedom matters: an in-state measurement has 2 dofs, an OOS one anywhere from
1 to 27, so a fixed threshold would silently become permissive for long tracks.
Non-finite distances are rejected rather than compared.

## Config

New `OOS` block in `cfg/oos.json` (copy of `sweep_dlt_nodesc.json` with
`use_OOS: true`), parsed into `OOSOptions`:

| key | default | meaning |
|---|---|---|
| `min_observations` | `OOS_update_min_observations` (5) | in-state views required |
| `max_observations` | `kMaxGroup` (15) | longer tracks thinned to this |
| `refine` | true | multi-view Gauss-Newton before the update |
| `max_iters`, `eps` | 10, 1e-5 | GN stopping |
| `Rtri` | 1.0 | triangulation weight |
| `max_mean_reproj_err` | 1.5 px | triangulation gate, per view |
| `zmin`, `zmax` | 0.05, 50 | depth gate |
| `MH_thresh` | 5.991 | Mahalanobis gate, per dof; <= 0 disables |

## Instrumentation

Per-step counters (candidates / used / too_short / bad_triangulation / gated /
rows) with run totals printed by `~Estimator` when `use_OOS` is on, plus the
observation-coverage diagnostic that turned out to be the whole story of this
milestone.

## Result: it runs, and it barely does anything

room1, seed 0:

```
candidates=4322  used=1  too_short=4231  bad_triangulation=3  gated=0
rows=9  observations=6
views/candidate: all=4.53  instate=0.42
instate-view histogram: 0:2691 1:1318 2:193 3:19 4:10 5:2 6:2 7+:0
```

**One** feature out of 4322 contributed a measurement, for a total of 9 rows over
a 30943-frame sequence. ATE moves from 0.1304 to 0.1267 on room1, which is noise,
not a result.

The cause is not the OOS code. It is that XIVO has no pose window. Groups enter
the state only as the *reference group of a feature being promoted in-state*
(`AddFeaturesWithInGroups` → `AddGroupToState(f->ref())`, or
`AddGroupOfFeatures`), and leave as soon as they lose their in-state reference
features. In-state groups are therefore a sparse, feature-driven set of about a
dozen poses scattered over the trajectory — not the last N frames. A dropped
feature's observations are spread over the frames it was tracked through, and 62%
of candidates have **zero** observations from a group that is in the state; the
mean is 0.42 of 4.53 views. There is essentially nothing for the marginalized
constraint to constrain.

A second, smaller problem is visible in the same numbers: candidates average only
4.53 observations *in total*, so even a perfect pose window would leave many
tracks below a `min_observations` of 5.

Both are M3's job:

1. maintain an explicit sliding pose window — augment the state with the current
   pose every frame (what the deleted MSCKF code did: `AddGroupToState(g)` for
   every new group, plus strided pruning with `oos_discard_step`), so that a
   dropped track's observations are all in-state;
2. give it room. `kMaxGroup` is 15 and the pose window would compete with the
   reference groups of in-state features for the same slots. `EKF_MAX_GROUPS` is
   a compile-time define (commented out in `src/CMakeLists.txt`), so the window
   length is a build-time knob for now;
3. reconsider `min_observations` — `2n-3` rows means `n = 2` already yields one
   usable row, so 3 or 4 is defensible once coverage is fixed.

Also, `EnforceMaxGroupLifetime` needs attention in M3: it removes groups that own
no reference features, with a debug `CHECK(!g->instate())`. Pose-window groups
own no features at all, so once they age past `max_group_lifetime` that check
would fire and, in a release build, a group would be dropped from the graph while
still holding a state slot.
