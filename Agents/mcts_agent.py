from Agents.mcts import calculate_move, MCTS
from Engine.state import State
from Eval.tourney import list_checkpoints
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]    # for children to get path to load / save weights



class MCTSAgent:
    """
    Wrapper class for all AI agents, who each use MCTS search.
    """
    CHECKPOINT_DIR = None       # each child sets for their own path
    ADAM_PATH = None
    RELEASE_WEIGHTS = None      # ^
    CREATE_BOARD = staticmethod(State.create_onehot_board)     # network input; children with a different input override it

    def __init__(self, load_checkpoint=True):
        self.model = self.build_model()
        self.mcts = MCTS()
        if load_checkpoint:
            self.load_weights()


    def build_model(self):
        """
        Return the ModelTree for this agent. Must be overwritten by child class
        """
        raise NotImplementedError


    def load_weights(self):
        """
        Loads final weights for model if published, or most recent checkpoint if not. Nothing happens if no checkpoints exist.
        """
        ckpts = list_checkpoints(self.CHECKPOINT_DIR)   # sorted by cycle number, not by name ("gen_1000" < "gen_999" as text)
        if ckpts:
            self.model.load(str(ckpts[-1][1]))          # for training
        elif self.RELEASE_WEIGHTS.exists():
            self.model.load(str(self.RELEASE_WEIGHTS))       # used once training is finished


    def get_move(self, state, training=False):
        return self.mcts.calculate_move(state, self.model, training, self.CREATE_BOARD)


    def analyze(self, state, num_simulations, progress):  
        """
        Search state without noise for up to num_simulations, calling progress(root, sims_done) as it goes
        (see calculate_move). Used by the UI's eval bar; the move it returns is ignored there.
        Sets keep_searching to true to find faster proven mates once proven mate is returned already.
        """
        return self.mcts.calculate_move(state, self.model, False, self.CREATE_BOARD, num_simulations=num_simulations, progress=progress, keep_searching=True, total_visits=True)


    def train(self, X, y, masks, return_loss=False):
        return self.model.fit(X, y, masks, batch_size=128, return_loss=return_loss)


    def evaluate(self, X, y, masks):
        return self.model.evaluate(X, y, masks)
    
