## Goal

Support bfloat16 (bf16) as the main numerical type (number_t) in the system without degrading the system accuracy (measured by ATE on the 6 TUM-VI sequences)
while improving the runtime efficiency.


## Definitions

- workspace directory: `/home/ubuntu/workspace/auto-slam-engineer`
- xivo package: `/home/ubuntu/workspace/auto-slam-engineer/xivo`
- tum-vi dataset: `/home/ubuntu/workspace/auto-slam-engineer/data/tumvi`
- notes directory `/home/ubuntu/workspace/auto-slam-engineer/notes-n-prompts`

## Requirements

1. Optimize the code base for two settings: 1/ monocular camera + IMU, and 2/ stereo camera + IMU. In both settings, fix the capacity of the state, and optimize the 
efficiency of the system for the fixed capacity.
2. Make a plan first on how to optimize the code base, and split the implementation into meaningful milestones. Write the plan to the notes directory.
3. Make sure each milestone is sufficiently tested. If possible, run end-to-end evaluation to measure the performance of the system when appropriate.
4. Commit each milestone as a git commit.
5. Once done, generate a report in the notes directory with name "report-bf16.md"
6. In the process of implementing and optimizing the code, and tuning the configurations, leave detailed notes under "notes-bf16" sub-directory in the notes directory.
7. When implementing and experimenting the feature, feel free to use sub-agents as needed. You can use git worktree to work on multiple features in parallel, but the final feature should be delivered in the auto-bf16 branch of the xivo package.
9. Please create a git worktree first in the workspace directory. The worktree is named "xivo-bf16", and it's created from the auto branch.


NOTE:
1. Use FPS as the metric to measure efficiency, and optimize the codebase to improve FPS.
2. Do NOT degrade the accuracy metrics (ATE and RPE).


## Exit criteria

1. After optimization, the code is much more efficient than it was.
2. The accuracy metrics do not degrade.
