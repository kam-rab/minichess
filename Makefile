QUEST = quest
REMOTE = ~/minichess

.PHONY: push pull submit status log

push:     # code + state to Quest
	rsync -avz --exclude .venv --exclude .git --exclude __pycache__ ./ $(QUEST):$(REMOTE)/
pull:     # results back from Quest
	rsync -avz $(QUEST):$(REMOTE)/Training/Checkpoints/leela/ Training/Checkpoints/leela/
	rsync -avz $(QUEST):$(REMOTE)/Eval/training/leela/ Eval/training/leela/
submit:
	ssh $(QUEST) 'cd $(REMOTE) && sbatch Training/train.sh'
status:
	ssh $(QUEST) 'squeue --me'
log:
	ssh $(QUEST) 'tail -f $$(ls -t $(REMOTE)/logs/*.out | head -1)'