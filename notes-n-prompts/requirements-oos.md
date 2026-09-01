## Goal
Make xivo support out-of-state (OOS, also known as MSCKF -- Multi-State Constrained Kalma Filter) update along with 
in-state update.

The current xivo package may already have some support of out-of-state update, but it may not work well. Please
check the implementation, improve it, and tune the system so that when out-of-state and in-state update are used together,
the system performance meets the exit criteria.

## Definitions

- workspace directory: `/home/ubuntu/workspace/auto-slam-engineer`
- xivo package: `/home/ubuntu/workspace/auto-slam-engineer/xivo`
- tum-vi dataset: `/home/ubuntu/workspace/auto-slam-engineer/data/tumvi`
- notes directory `/home/ubuntu/workspace/auto-slam-engineer/notes-n-prompts`

## Requirements

0. Assume monocular + IMU setting.
1. Implement code in the xivo package to support out-of-state update.
2. The implemented out-of-state update, when used together with in-state feature update, performs much better than the the performance reported in README.md under workspace directory.
3. Make a plan first on how to implement the feature, and split the implementation into meaningful milestones. Write the plan to the notes directory.
4. Make sure each milestone is sufficiently tested. If possible, run end-to-end evaluation to measure the performance of the system when appropriate.
5. Commit each milestone as a git commit.
6. Once done, generate a report in the notes directory with name "report-oos.md"
7. In the process of implementing and optimizing the code, and tuning the configurations, leave detailed notes under "notes-oos" sub-directory in the notes directory.
8. When implementing and experimenting the feature, feel free to use sub-agents as needed. You can use git worktree to work on multiple features in parallel, but the final feature should be delivered in the auto-oos branch of the xivo package.
9. Please create a git worktree first in the workspace directory. The worktree is named "xivo-oos", and it's created from the auto branch.


## Exit criteria

1. The implementation, in monocular + IMU setting, has a mean ATE less than 0.06 meters on the six (room1 to room6) sequences in tum-vi.
2. If possible, the mean ATE on the six sequences should be as small as possible.
3. The mean RPE metric on the six tum-vi sequences, in the monocular + IMU setting, is less than 0.5 degrees.
