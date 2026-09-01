# M6 — the memory pools, and the abort that hid behind them

## Symptom

Five of twelve runs in a capacity sweep (`sweeps/m6f.log`, arms `f120t240` and
`f150t300`) produced a ground-truth file and no trajectory file. The harness piped
stderr into a `grep`, so the cause was swallowed; re-running one by hand gave:

    F20260822 02:32:38 mm.cpp:94] Out of feature slots in the memory manager
      @ xivo::Tracker::UpdateLK()
      @ xivo::Tracker::UpdateStereo()
      @ xivo::Estimator::VisualMeasStereoInternal()

Ten minutes into room1, from inside the tracker, with no mention of a config key.

## Cause

`memory.max_features` / `memory.max_groups` size fixed pre-allocated pools
(`src/mm.h`). `CircBufWithHash::GetItem` hands out a slot, and when every slot is
active it calls `LOG(FATAL)`. There is no growth path and no backpressure.

The sweep raised `tracker_cfg.num_features_max` to 240 and 300 while leaving the
pool at 400. Upstream ships 200 features for a tracker cap of 60 — a 3.3x margin —
so nobody had run into the ceiling before. My arms were at 1.7x and 1.3x.

This is a config error in the sweep, not an xivo defect. But the *diagnosis* cost
an hour, which is a defect in the diagnostics.

## How much margin is actually needed

A slot is held for every feature the tracker is tracking **plus** every feature
the tracker has dropped that the estimator has not destroyed yet. So the tracker
cap is a hard lower bound but not a sufficient one. Instrumenting the pool with a
high-water mark (`CircBufWithHash::peak_active`) and running room6 at a tracker cap
of 300:

    E mm.cpp:50] MemoryManager: feature pool 360/400 active; raise
                 memory.max_features -- running out is fatal

360 active slots for a 300-feature tracker, reached within 30 s. So peak usage
runs about **1.7x the tracker cap**, which is consistent with the arm at 1.67x
aborting and the arm at 2.2x surviving all six rooms.

## What was added

1. **`CheckMemoryPools`** (`src/factory.cpp`), called from both `CreateSystem` and
   `CreateSystemTrackerOnly` right after the tracker exists. Fatal, at startup,
   naming both knobs, when the feature pool is below the tracker cap or below the
   EKF's compile-time capacity, or the group pool is below the EKF's group
   capacity. Advisory below 2x the tracker cap.
2. **A 90%-full high-water warning** in `CircBufWithHash::NoteActivation`, for the
   case the up-front check cannot catch (a pool that is nominally large enough but
   fills anyway).
3. `src/test/unittest_memory_pools.cpp` — five death tests, including one that
   asserts the *shipped* config passes its own check, so the guard cannot be
   tightened into breaking every run.

Both messages are `LOG(ERROR)`, not `LOG(WARNING)`, deliberately: glog's default
`stderrthreshold` is ERROR, so a WARNING goes only to the log files. The first
version of the advisory was a WARNING and was invisible in exactly the situation it
existed for.

## Shipped values

`cfg/tumvi_stereo.json` runs a tracker cap of 180 with a pool of 800 (4.4x) and
300 groups. The pool is pre-allocated `Feature` objects; oversizing it costs a
one-time allocation and nothing per frame, so there is no reason to run it close.

See [[m6-capacity]] for why the tracker cap is 180 in the first place.
