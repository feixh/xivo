## Goal

Scan the code and fix bugs.



## Definitions

- workspace directory: `/home/ubuntu/workspace/auto-slam-engineer`
- xivo package: `/home/ubuntu/workspace/auto-slam-engineer/xivo`
- tum-vi dataset: `/home/ubuntu/workspace/auto-slam-engineer/data/tumvi`
- notes directory `/home/ubuntu/workspace/auto-slam-engineer/notes-n-prompts`

## Requirements

1. Assume the monocular camera + IMU setting.
2. Make a plan first on how to find and fix the bugs, and split the implementation into meaningful milestones. Write the plan to the notes directory.
3. Make sure each milestone is sufficiently tested. If possible, run end-to-end evaluation to measure the performance of the system when appropriate.
4. Commit each milestone as a git commit.
5. Once done, generate a report in the notes directory with name "report-bugfix.md"
6. In the process of implementing and optimizing the code, and tuning the configurations, leave detailed notes under "notes-bugfix" sub-directory in the notes directory.
7. When implementing and experimenting the feature, feel free to use sub-agents as needed. You can use git worktree to work on multiple features in parallel, but the final feature should be delivered in the auto-bugfix branch of the xivo package.
9. Please create a git worktree first in the workspace directory. The worktree is named "xivo-bugfix", and it's created from the auto branch.




## Exit criteria

The code is free of bugs.

## Stretch goal: Performance improvement
1. If possible, the implementation, in monocular + IMU setting, has a mean ATE less than 0.06 meters on the six (room1 to room6) sequences in tum-vi.
2. If possible, the mean ATE on the six sequences should be as small as possible.
3. If possible, the mean RPE metric on the six tum-vi sequences, in the monocular + IMU setting, is less than 0.5 degrees.
