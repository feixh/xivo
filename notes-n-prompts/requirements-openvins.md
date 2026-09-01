## Goal
Take a look at OpenVINS implementation https://github.com/rpng/open_vins and borrow algorithmic ideas from it to enhance 
xivo



## Definitions

- workspace directory: `/home/ubuntu/workspace/auto-slam-engineer`
- xivo package: `/home/ubuntu/workspace/auto-slam-engineer/xivo`
- tum-vi dataset: `/home/ubuntu/workspace/auto-slam-engineer/data/tumvi`
- notes directory `/home/ubuntu/workspace/auto-slam-engineer/notes-n-prompts`

## Requirements

0. Please stick to the monocular + IMU setting. We can work on stereo + IMU and other sensor settings separately.
1. Don't borrow MSCKF or out-of-state feature update ideas, we will work on this separately.
2. Make a plan first on how to implement the feature, and split the implementation into meaningful milestones. Write the plan to the notes directory.
3. Make sure each milestone is sufficiently tested. If possible, run end-to-end evaluation to measure the performance of the system when appropriate.
4. Commit each milestone as a git commit.
5. Once done, generate a report in the notes directory with name "report-openvins.md"
6. In the process of implementing and optimizing the code, and tuning the configurations, leave detailed notes under "notes-openvins" sub-directory in the notes directory.
7. When implementing and experimenting the feature, feel free to use sub-agents as needed. You can use git worktree to work on multiple features in parallel, but the final feature should be delivered in the auto-openvins branch of the xivo package.
9. Please create a git worktree first in the workspace directory. The worktree is named "xivo-openvins", and it's created from the auto branch.


IMPORTANT: Please do NOT try to build OpenVINS. Just downloading its code and browsing its code is Okay.


## Exit criteria

1. The implementation, in monocular + IMU setting, has a mean ATE less than 0.06 meters on the six (room1 to room6) sequences in tum-vi.
2. If possible, the mean ATE on the six sequences should be as small as possible.
3. The mean RPE metric on the six tum-vi sequences, in the monocular + IMU setting, is less than 0.5 degrees.
