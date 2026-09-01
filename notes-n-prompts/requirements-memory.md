## Goal

Use tools like AddressSanitizer and LeakSanitizer to find potential memory leaks and fix them.


## Definitions

- workspace directory: `/home/ubuntu/workspace/auto-slam-engineer`
- xivo package: `/home/ubuntu/workspace/auto-slam-engineer/xivo`
- tum-vi dataset: `/home/ubuntu/workspace/auto-slam-engineer/data/tumvi`
- notes directory `/home/ubuntu/workspace/auto-slam-engineer/notes-n-prompts`

## Requirements

1. Assume the monocular camera + IMU setting.
2. Make a plan first on how to find and fix the memory leaks, and split the implementation into meaningful milestones. Write the plan to the notes directory.
3. Make sure each milestone is sufficiently tested. If possible, run end-to-end evaluation to measure the performance of the system when appropriate.
4. Commit each milestone as a git commit.
5. Once done, generate a report in the notes directory with name "report-memory.md"
6. In the process of implementing and optimizing the code, and tuning the configurations, leave detailed notes under "notes-memory" sub-directory in the notes directory.
7. When implementing and experimenting the feature, feel free to use sub-agents as needed. You can use git worktree to work on multiple features in parallel, but the final feature should be delivered in the auto-memory branch of the xivo package.
9. Please create a git worktree first in the workspace directory. The worktree is named "xivo-memory", and it's created from the auto branch.

NOTE:
1. Implementing new algorithmic features is out of scope for this task.
2. Tuning the system and improving its performance is out of scope.




## Exit criteria

The code is free of memory leaks.
And the performance of the system, on monocular + IMU setting, does not regress.

