from Agents.mcts_agent import MCTSAgent, REPO_ROOT
from Model.model import *
from Model.layers import *
from Model.loss import *
from Model.regularization import *
from Engine.state import State
from pathlib import Path


class Agent(MCTSAgent):
    """
    A policy + value ModelTree with l2 regularization.
    Leela also has input nodes for repetition count and move number. 
    Lastly, it includes a third output head that predicts move_number: the number of moves left in the game.
    Used to favor winning earlier.
    """
    CHECKPOINT_DIR = REPO_ROOT / "Training" / "Checkpoints" / "leela"
    RELEASE_WEIGHTS = Path(__file__).resolve().parent / "Weights" / "leela.npz"
    ADAM_PATH = REPO_ROOT / "Training" / "Checkpoints" / "leela" / "Adam.npz"
    CREATE_BOARD = staticmethod(State.create_leela_input)     # board + move counter + repetition flag (302 inputs)


    def build_model(self):
        LEARNING_RATE = 0.001       # lowered to 1e-3 because models use Adam now
        ALPHA = 0.01                # regularizer for l2 
        HEAD_WEIGHTS = (1, 1, 0.5)  # controls how much each head's loss function affects shared trunk: (policy, value, moves_left)

        trunk = Model(
            layers= [
                FullyConnected(302, 256, regularizer= Regularizer(alpha=ALPHA)),
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

        moves_left = Model(
            layers= [
                FullyConnected(256, 64, regularizer= Regularizer(alpha=ALPHA)),
                ReluActivation(),
                FullyConnected(64, 1, regularizer= Regularizer(alpha=ALPHA)), 
                SigmoidActivation()], 
            loss= SquaredLoss(),
            learning_rate= LEARNING_RATE,
        )

        trunk_node = Node(model=trunk, loss_weights=HEAD_WEIGHTS)
        policy_node = Node(parents=[trunk_node], model=policy)
        value_node = Node(parents=[trunk_node], model=value)
        moves_left_node = Node(parents=[trunk_node], model=moves_left)
        trunk_node.children = [policy_node, value_node, moves_left_node]

        return ModelTree(roots=(trunk_node,), leaves=(policy_node, value_node, moves_left_node))
