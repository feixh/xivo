## Goal
Make xivo support stereo cameras.

## Definitions

- workspace directory: `/home/ubuntu/workspace/auto-slam-engineer`
- xivo package: `/home/ubuntu/workspace/auto-slam-engineer/xivo`
- tum-vi dataset: `/home/ubuntu/workspace/auto-slam-engineer/data/tumvi`
- notes directory `/home/ubuntu/workspace/auto-slam-engineer/notes-n-prompts`

## Requirements

1. Implement code in the xivo package to support stereo cameras + IMU for visual inertial odometry.
2. The implemented stereo + IMU odometry should perform much better than the monocular + IMU odometry whose performance is reported in README.md under workspace directory.
3. Make a plan first on how to implement the feature, and split the implementation into meaningful milestones. Write the plan to the notes directory.
4. Make sure each milestone is sufficiently tested. If possible, run end-to-end evaluation to measure the performance of the system when appropriate.
5. Commit each milestone as a git commit.
6. Once done, generate a report in the notes directory with name "report-stereo.md"
7. In the process of implementing and optimizing the code, and tuning the configurations, leave detailed notes under "notes-stereo" sub-directory in the notes directory.
8. When implementing and experimenting the feature, feel free to use sub-agents as needed. You can use git worktree to work on multiple features in parallel, but the final feature should be delivered in the auto-stereo branch of the xivo package.


## Exit criteria

1. The implemented stereo + IMU odometry has a mean ATE less than 0.06 meters on the six (room1 to room6) sequences in tum-vi.
2. If possible, the mean ATE on the six sequences should be as small as possible.
3. The mean RPE metric on the six tum-vi sequences, in the stereo + IMU setting, is less than 0.5 degrees.
