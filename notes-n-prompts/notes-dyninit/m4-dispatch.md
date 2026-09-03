# M4 -- choosing the initializer, and holding the messages while it is chosen

M0-M3 built the pieces: a detector that says static or dynamic (M1), a linear
solve for velocity and gravity (M2), and a bundle adjustment that refines both
plus the biases (M3). None of them was wired into the filter. M4 wires them in.

| file | what it is |
|---|---|
| `src/init_dispatch.{h,cpp}` | the dispatcher: accumulate, decide, report |
| `src/estimator.{h,cpp}` | the divert, the replay, and the handoff into `X_` |
| `cfg/euroc_{mono,stereo}.json` | the `dynamic_init` block, shipped **off** |
| `harness/m4_bitident.sh` | the gate below |

`ctest`: **26/26 green.**

## The shape of it

The detector needs about a second of images (`init_detect.h` explains why nothing
cheaper separates the classes), but XIVO's static initializer is ready after 20
accelerometer samples -- 0.1 s. Something has to give, and what gives is latency:

1. While the dispatcher is deciding, **every** message is diverted into
   `init_buf_` and the estimator is not called at all. Not a counter, not a
   clock, not the accelerometer buffer.
2. On a **static** verdict the buffer is replayed in order. The estimator never
   saw those messages the first time, so it reaches the state it would have
   reached with the whole feature deleted.
3. On a **dynamic** verdict Stage A and Stage B solve the window, `X_.Rsg`,
   `X_.Vsb`, `X_.bg`, `X_.ba` and the propagation clock are seeded from the
   result, and only the messages *after* the window's last frame are replayed.
   The earlier ones are not discarded -- their information is what the bundle
   adjustment just consumed.

The interception point is `Estimator::Dispatch`, called from the one place the
message heap is drained (`MaintainBuffer` on the synchronous path, the worker
lambda on the asynchronous one). Diverting there rather than at the public
`VisualMeas`/`InertialMeas` entry points matters: the heap is what sorts images
against IMU samples on the temporally-corrected clock, so a message reaches
`Dispatch` in exactly the order the filter would have executed it, and the replay
is that same order.

## The gate, and why it is not `cmp`

The prediction M4 was supposed to live or die by, from `plan-dyninit.md`:

> with `dynamic_init` on, **the nine static EuRoC sequences must be bit-identical
> to the current shipped result**, because the detector selects the static path
> and the static path is untouched; only MH_01 and MH_02 may change.

The premise is right and the test as written is unachievable. Two things differ
on a static sequence, and **neither of them is the buffering**:

**1. The init-window poses are never reported.** The filter has not started, so
there is nothing to report. `pyxivo.py` writes a pose per visual call gated on
`VisionInitialized()`, so `on` is missing a prefix of `off` -- 18 poses on
V1_01_easy, the 0.9 s the detector took. Comparison has to align on timestamp,
not on line number. This is the latency of step 1 made visible; OpenVINS'
initializer has the same property.

**2. Online temporal calibration.** `estimator.cpp:1664`, `VisualMeas`:

```cpp
timestamp_t ts{ts_raw};
#ifdef USE_ONLINE_TEMPORAL_CALIB
  if (X_.td >= 0) ts += timestamp_t(uint64_t(X_.td * 1e9));
```

The *enqueue* timestamp of an image carries the current `X_.td`, because the heap
has to sort it against IMU samples on the corrected clock. Inside the init window
no EKF update has run, so every buffered frame is stamped with `td_0`; the
unbuffered filter had already moved `td` a few hundred nanoseconds by frame 5.

That difference is sub-microsecond and it is enough. V1_01_easy, first 301
executed messages, `off` against `on` -- the image timestamps drift apart and the
image/IMU tie in the heap flips:

```
off: 122 img 1403715273.8121428     on: 122 imu 1403715273.8121431
off: 123 imu 1403715273.8121431     on: 123 img 1403715273.8121431
```

One consequence is visible directly in the logs: `Propagate` warns
`measurement timestamps coincide?` once in the baseline and **18 times** with the
divert -- once per replayed frame, because each image now lands on top of an IMU
sample it used to sort ahead of. From there the two runs diverge chaotically:
1e-4 m at the handoff, 0.06 m at the end of a 145 s trajectory.

Note which value is the *defensible* one. Under the divert the filter genuinely
has no temporal-calibration estimate yet when those frames are enqueued, so
`td_0` is the only thing it could honestly use. The baseline's value is not more
correct, just unreproducible once the frames are held back.

### So the gate is: freeze `td` and demand exactness

Set `P.td = 0` -- zero prior variance, so the EKF can never move `td` and every
frame is enqueued with the same offset in both runs -- and the residual has to be
zero. On V1_01_easy:

```
off 2909 poses, on 2891 | offset 18 | aligned tail 2891 vs 2891 | mismatches: 0
```

**Every shared pose byte-identical.** That is the real test of divert-and-replay,
and it is exact rather than argued: a single wrong message order, a single
mutated pixel, one extra draw from a shared counter, and it fails.

It also confirms the message-count argument. `MaintainBuffer` pops exactly one
message per message pushed, so at the instant of the decision the replay has
executed exactly as many messages as the unbuffered filter would have, in the
same order -- the two are *in step* from the handoff onward, not merely close.
Had the burst put the estimator ahead or behind, the aligned tails would have had
different lengths.

`m4_bitident.sh` runs both variants: the frozen-`td` pair as a pass/fail gate,
and the shipped pair as a measured report.

## Results

```
./notes-n-prompts/notes-dyninit/harness/m4_bitident.sh --profile euroc_mav
./notes-n-prompts/notes-dyninit/harness/m4_bitident.sh --profile tumvi_room \
    --dynamic "" --no-ship
```

### The gate: frozen `td`, 11 EuRoC sequences x 2 modes

**18 of 18 static rows exact.** Nine sequences, mono and stereo, every shared
pose byte-identical:

```
MH_03_medium    mono    static   exact   2679 shared poses, on starts 16 poses later
MH_04_difficult mono    static   exact   2011 shared poses, on starts 17 poses later
MH_05_difficult mono    static   exact   2251 shared poses, on starts 18 poses later
V1_01_easy      mono    static   exact   2891 shared poses, on starts 18 poses later
V1_02_medium    mono    static   exact   1689 shared poses, on starts 18 poses later
V1_03_difficult mono    static   exact   2128 shared poses, on starts 18 poses later
V2_01_easy      mono    static   exact   2259 shared poses, on starts 15 poses later
V2_02_medium    mono    static   exact   2327 shared poses, on starts 15 poses later
V2_03_difficult mono    static   exact   1900 shared poses, on starts 10 poses later
... stereo identical, 9 for 9
```

And the two the detector routes to the BA are the two M1 predicted, before any of
this was wired up:

```
MH_01_easy      mono    dynamic  CHANGED  on starts 11 poses (0.55 s) EARLIER, first at +0 by 0.2255 m
MH_02_easy      mono    dynamic  CHANGED  on starts 11 poses later, first at +0 by 0.0904 m
```

`MH_01_easy` starts reporting **0.55 s earlier** with the dynamic branch, in both
modes. The static path has to wait out a wrong initial velocity before
`VisionInitialized()` turns true; seeding from the BA skips that wait. That is the
feature working, visible in the one place a latency change shows up.

The 10-18 dropped poses are the init window: 0.5-0.9 s at 20 Hz, exactly what
step 1 costs.

### TUM-VI room1-6: a regression guard

All six rooms start static, so nothing should reach the BA, and nothing does:
**12 of 12 rows exact** (six rooms x two modes). This is the check that the
dispatcher does not fire on a dataset it was never tuned for.

### The shipped config, n=1: what the 300 ns is worth

ATE RMSE [m], `evaluate_ate.py`, 0.02 s window, one seed:

| | mean off | mean on | delta |
|---|---|---|---|
| stereo | 0.0976 | 0.1056 | +0.0080 |
| mono | 0.1754 | 0.1890 | +0.0136 |

Split by branch:

| | MH_01 (dynamic) | MH_02 (dynamic) | 9 static, mean \|delta\| | 9 static, max \|delta\| |
|---|---|---|---|---|
| stereo | **-0.0121** | +0.0276 | 0.0088 | 0.0231 |
| mono | **-0.0111** | +0.0234 | 0.0235 | 0.0713 |

Read the third and fourth columns first, because they are the ones that say how
to read the rest. Those nine sequences run **provably identical arithmetic** --
the frozen-`td` gate just proved it to the last digit -- and their ATE still moves
by 0.009 m on average and 0.071 m at worst. All of that is the chaotic
amplification of a 300 ns change in when an image sorts against an IMU sample.

So this table's `mean delta` column measures almost nothing about dynamic
initialization, and the +0.008 / +0.014 must not be read as a regression. What it
does measure is XIVO's **single-run ATE noise floor on EuRoC**: ~0.009 m stereo
and ~0.024 m mono, with a mono tail past 0.07 m. That is an order of magnitude
above the +/-0.007 sd measured on TUM-VI room1, and it is why M5's evaluation is
n=10 with a jitter ensemble rather than a pair of runs.

Against that floor, MH_01 improving by 0.012 m in both modes is suggestive and
nothing more, and MH_02 worsening by ~0.025 m in both modes is equally
suggestive in the other direction -- both inside the noise a single run carries,
both worth resolving with an ensemble. That is M5's job, and M4 is not the
milestone that gets to claim an accuracy result.

Cost, n=1, from the same pass: throughput 73.5 -> 72.9 FPS stereo and
123.1 -> 120.8 mono (-0.8% / -1.9%, the detector's extra KLT during the window),
peak RSS 130.7 -> 138.8 MB stereo and 119.5 -> 126.3 MB mono (+8 MB / +7 MB, the
buffered messages and the window's tracks). Both are one-off startup costs that
end when the verdict lands, and both need the proper protocol -- one core, ASLR
off, `-mode runOnly` -- before they mean anything.

## Design decisions worth recording

**Where the biases go.** `init_preint.h` and `ComposeMotion` use the identical
convention, `Cg*w - bg` and `Ca*a - ba`, so `X_.bg = state.bg` is a direct
assignment with no sign or frame change. The dispatcher pre-multiplies by
`imu_.Cg()` / `imu_.Ca()` before feeding the window, so the window sees what the
filter's propagation sees. `clamp_signals_` is deliberately *not* applied: it
guards the filter's propagation, and the replay still passes through it.

**Handoff at the window's last frame.** That is the instant the filter is about
to declare `Rsb = I`, so it is the only frame whose body coordinates the filter's
state means anything in. `Rsg` is built by `FromTwoVectors(-g_, -gravity_body)`
with the third component of the logarithm dropped, mirroring the static path at
`estimator.cpp:777-781` -- so both paths agree on the yaw convention rather than
each picking their own.

**`Vsb` is spatial, not body.** `ComposeMotion` integrates `Tsb += V*dt`, so the
decision's body-frame velocity is rotated: `X_.Vsb = X_.Rsb * d.Vsb`. `Rsb` is
the identity on every shipped config -- the same assumption the static path's
`Rsg` construction makes -- but rotating explicitly costs nothing and keeps this
correct if that stops being true.

**Rejections, not repairs.** A dynamic solve that fails any of its gates
(non-finite state, reprojection median above `max_pixel_median`, speed above
`max_speed`) hands the problem back to the static path. The static path is merely
wrong about the initial velocity; a half-converged BA is wrong about the
geometry, which is worse. Same reasoning for a dynamic verdict the window cannot
act on (`num_frames < min_frames`): fall back, because that failure is the one the
filter already handled before this code existed.

**The verdict is cached, and the detector then stops eating frames.** Once
`Classify()` has answered, `AddImage` no longer feeds the detector. It saves a
KLT per frame, but the real reason is that it makes the verdict a function of a
fixed prefix of the data instead of something that can flip half way through
filling the window -- which would leave the filter's start depending on exactly
when `Decide()` happened to be called.

## A tooling fix this needed

Every initialization summary in XIVO is a `LOG(INFO)`, and `CMakeLists.txt` had
`add_definitions(-DGOOGLE_STRIP_LOG=1)` unconditionally -- the call sites are
*compiled out*, so no amount of `GLOG_minloglevel=0` brings them back, in
`pyxivo` or in `bin/vio`. Diagnosing the above needed them. That line is now

```cmake
set(XIVO_STRIP_LOG "1" CACHE STRING "...")
add_definitions(-DGOOGLE_STRIP_LOG=${XIVO_STRIP_LOG})
```

so `cmake -DXIVO_STRIP_LOG=0 .` in `build/` restores gravity init, the loaded
calibration, and the dispatch decision. The default is unchanged.
