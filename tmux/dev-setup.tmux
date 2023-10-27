tmux new-session -s aaml -n Main -d
tmux new-window -n Scheduler -t aaml -d
tmux new-window -n Deploy-sim -t aaml -d
tmux new-window -n Pod-sim -t aaml -d
tmux new-window -n Monitor -t aaml -d

