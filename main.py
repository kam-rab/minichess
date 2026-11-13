# for self play workers, needs to be set before numpy is imported
import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from Training.loop import *
from Agents.AIs.unregularized import Agent as unregularized
from Agents.AIs.l2 import Agent as l2
from Agents.AIs.leela import Agent as leela

if __name__=="__main__":
    train(leela, resume_training=True, num_cycles=1001)
    

"""
To watch evals vs greedy in real time, run:
python -m Eval.plot_results --agent AGENT_NAME --watch 

To watch loss in real time, run:
python -m Eval.plot_results --agent AGENT_NAME --loss --watch 

To watch game log stats in real time, run:
python -m Eval.plot_results --agent AGENT_NAME --games --watch

To watch conversion scatters in real time, run:
python -m Eval.plot_results --agent AGENT_NAME --material --watch --last 100 
"""

