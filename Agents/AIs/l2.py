from Agents.mcts_agent import MCTSAgent, REPO_ROOT
from Model.model import *
from Model.layers import *
from Model.loss import *
from Model.regularization import *
from pathlib import Path


class Agent(MCTSAgent):
    """
    A policy + value ModelTree with l2 regularization
    """
    CHECKPOINT_DIR = REPO_ROOT / "Training" / "Checkpoints" / "l2"
    RELEASE_WEIGHTS = Path(__file__).resolve().parent / "Weights" / "l2.npz"


    def build_model(self):
        LEARNING_RATE = 0.001       # lowered to 1e-3 because models use Adam now
        ALPHA = 0.01        # regularizer for l2 

        trunk = Model(
            layers= [
                FullyConnected(300, 256, regularizer= Regularizer(alpha=ALPHA)),
                ReluActivation(),
                FullyConnected(256, 256, regularizer= Regularizer(alpha=ALPHA)),
                ReluActivation(),
                FullyConnected(256, 256, regularizer= Regularizer(alpha=ALPHA)),
                ReluActivation()], 
            learning_rate= LEARNING_RATE
            )

        value = Model(
            layers=[
                FullyConnected(256, 64, regularizer= Regularizer(alpha=ALPHA)), 
                ReluActivation(), 
                FullyConnected(64, 1, regularizer= Regularizer(alpha=ALPHA)), 
                TanhActivation()],
            loss= SquaredLoss(), 
            learning_rate= LEARNING_RATE
            )
        
        policy = Model(
            layers= [
                FullyConnected(256, 625, regularizer= Regularizer(alpha=ALPHA)), 
                SoftmaxActivation()],
            loss= CrossEntropyLoss(),
            learning_rate= LEARNING_RATE, 
            softmax_ce_block= True,
            apply_mask=True)

        trunk_node = Node(model=trunk)
        policy_node = Node(parents=[trunk_node], model=policy)
        value_node = Node(parents=[trunk_node], model=value)
        trunk_node.children = [policy_node, value_node]

        return ModelTree(roots=(trunk_node,), leaves=(policy_node, value_node))
