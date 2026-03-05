# Task Summary

1. SSH into dev-container and work under /workspace/verl.
2. Activate virtual environment with source ./env/bin/activate.
3. Execute ./run_flowgrpo.sh inside a tmux session (to survive unstable SSH).
4. Wait until execution gets stuck before entering the training loop.
5. Locate where/why it hangs, fix the bug, and rerun.
6. Repeat until the job successfully enters the training loop.
7. After each run, clear GPU jobs with ./scripts/kill_all_gpu_procs.sh.
8. Do not finalize until training loop entry is confirmed.
9. Keep this summary in both:
   - local repo: ./task.md
   - remote: dev-container:/workspace/verl/.codex/task.md
