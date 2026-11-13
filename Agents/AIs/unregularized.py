from Agents.mcts_agent import MCTSAgent, REPO_ROOT
from Model.model import *
from Model.layers import *
from Model.loss import *
from pathlib import Path


class Agent(MCTSAgent):
    """
    A policy + value ModelTree without regularization or dropout
    """
    CHECKPOINT_DIR = REPO_ROOT / "Training" / "Checkpoints" / "unregularized"
    RELEASE_WEIGHTS = Path(__file__).resolve().parent / "Weights" / "unregularized.npz"


    def build_model(self):
        LEARNING_RATE = 0.001

        trunk = Model(
            layers= [FullyConnected(300, 256),
                ReluActivation(),
                FullyConnected(256, 256),
                ReluActivation(),
                FullyConnected(256, 256),
                ReluActivation()], 
            learning_rate= LEARNING_RATE
            )

        value = Model(
            layers=[FullyConnected(256, 64), ReluActivation(), FullyConnected(64, 1), TanhActivation()],
            loss= SquaredLoss(), 
            learning_rate= LEARNING_RATE
            )
        
        policy = Model(
            layers= [FullyConnected(256, 625), SoftmaxActivation()],
            loss= CrossEntropyLoss(),
            learning_rate= LEARNING_RATE, 
            softmax_ce_block= True,
            apply_mask=True)

        trunk_node = Node(model=trunk)
        policy_node = Node(parents=[trunk_node], model=policy)
        value_node = Node(parents=[trunk_node], model=value)
        trunk_node.children = [policy_node, value_node]

        return ModelTree(roots=(trunk_node,), leaves=(policy_node, value_node))
