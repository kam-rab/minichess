from Agents.mcts import calculate_move, MCTS
from Engine.state import State
from Eval.tourney import list_checkpoints
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]    # for children to get path to load / save weights



class MCTSAgent:
    """
    Wrapper class for all AI agents, who each use MCTS search.
    """
    CHECKPOINT_DIR = None       # each child sets for their own path
    ADAM_PATH = None
    RELEASE_WEIGHTS = None      # ^
    PREFER_AVG = False          # load_weights takes the latest averaged checkpoint first; the UI turns it on, training never does
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
        With PREFER_AVG (set by the UI), the latest averaged checkpoint is loaded first if there is one.
        """
        if self.PREFER_AVG:
            avg = self.latest_avg()
            if avg is not None:
                print(f"{type(self).__module__}: loading {avg.name}")
                self.model.load(str(avg))
                return
        ckpts = list_checkpoints(self.CHECKPOINT_DIR)   # sorted by cycle number, not by name ("gen_1000" < "gen_999" as text)
        if ckpts:
            self.model.load(str(ckpts[-1][1]))          # for training
        elif self.RELEASE_WEIGHTS.exists():
            self.model.load(str(self.RELEASE_WEIGHTS))       # used once training is finished


    def latest_avg(self):
        """Path of the averaged checkpoint (<checkpoint dir>_avg/avg{n}_{gen}.npz) with the highest gen, the most
        checkpoints averaged on a tie, or None if there are none."""
        if self.CHECKPOINT_DIR is None:
            return None
        avgs = []
        for p in (self.CHECKPOINT_DIR.parent / f"{self.CHECKPOINT_DIR.name}_avg").glob("avg*_*.npz"):
            m = re.fullmatch(r"avg(\d+)_(\d+)\.npz", p.name)
            if m:
                avgs.append((int(m.group(2)), int(m.group(1)), p))
        return max(avgs)[2] if avgs else None


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
    
