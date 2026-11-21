import numpy as np
from Engine.state import State, BOARD_SIZE
from Model.util import encode, decode, create_mask, terminal_value

NUM_SIMULATIONS = 256      # adjust for early vs late training for better data
NOISE_ALPHA = 1        # controls shape of dirilecht noise added in training
NOISE_EPSILON = 0.25     # controls weight of noise 

T_START, T_MIN, T_HALFLIFE = 1.0, 0.15, 40     # temperature halves every T_HALFLIFE plies, down to T_MIN
VISIT_CUTOFF = 0.05     # lower bound on visit rate for sample-able moves

EXPLORATION_CONSTANT = 1.5

WIN_DECAY = 0.00         # simpler version of reducing num_moves

PROGRESS_EVERY = 25          # simulations between progress(root, sims) calls, if a progress function is passed

MOVES_LEFT_THRESHOLD = 0.8   # only when the side to move is clearly winning / losing
MOVES_LEFT_SCALE = 0.3       # k: bonus per 1.0 of (M_child - M_parent), M in units of plies/160
MOVES_LEFT_CLIP = 0.2        # c: largest |M_child - M_parent| that counts (~32 plies)


class Node: 
    def __init__(self, moves, policy_evals, state, color=None, terminal=None):
        """
        Tree structure for MCTS. Creates:
        - self.state: string game_state representation. Recover full state with state.from_string()
        - self.moves: list of legal moves (logits)
        - self.policy_evals: np array of policy network's output (same length as self.moves)
        - self.children_visits: np array of num visits to each child (same length as self.moves)
        - self.children_values: np array of running total terminal scores for each action taken from this node (same length as self.moves)
        - self.children: list of child Nodes (same length as self.moves)
        - self.terminal: flag for tree search: None if not terminal, otherwise 1/0/-1 holding outcome of game
        - self.color: +1 if white, -1 if black
        
        moves, policy_evals, children_visits, children_values, and children are all parallel
        Q(s, a) = self.children_values / self.children_visits
        U(s, a) = EXPLORATION_CONSTANT * self.policy_evals * (np.sum(self.children_visits)**(1/2)) / self.children_visits
        """

        self.state = state
        self.moves = moves
        self.policy_evals = policy_evals
        self.terminal = terminal

        self.children_visits = np.zeros_like(policy_evals, dtype=np.int64)
        self.children_values = np.zeros_like(policy_evals, dtype=np.float64)
        self.children_moves_left = np.zeros_like(policy_evals, dtype=np.float64)
        self.children = [None] * len(moves)
        self.proven = None     # proven mate idea: side to move to this node wins forcably (+1) or loses forcably (-1)
        self.proven_plies = None    # how far away mate is
        if color:
            self.color = color
        else:
            i = state.index('!')
            self.color = 1 if self.state[i+1:i+2] == 'w' else -1



class MCTS:
    def __init__(self, root=None):
        self.root = root
        self.rng = np.random.default_rng()


    def reset(self):
        """ Reset the tree at the end of a game"""
        self.root = None


    def calculate_move(self, state, model, training=False, create_board=State.create_onehot_board, num_simulations=None, progress=None, keep_searching=False, total_visits=False):
        """
        Run MCTS loop for NUM_SIMULATIONS iters (or num_simulations, if given). 
        progress: optional function called as progress(root, sims_done) after the first simulation and then every
        PROGRESS_EVERY simulations (used by the UI's eval bar). If it returns False, the search stops early.
        Sample move with probability defined by Temperature.
        Training = True adds Dirichlet noise, only turned off for evals.
        
        model must include model.evaluate(state.create_onehot_board())) which returns (moves, P, V):
        - legal moves (list of logits)
        - policy output (np array of size=len(moves))
        - value output (scalar)

        The MCTS loop does:
        - save tree for reuse (self.root)
        - find proven forced mates and return
        - rerun to find faster proven forced mate with keep_searching=True

        """
        X = create_board(state)     # to be returned if training

        # find already existing tree
        root = None
        if self.root is not None:
            curr = state.state_to_string()
            if self.root.state == curr:
                root = self.root
            else:
                for child in self.root.children:
                    if child is None:
                        continue
                    if child.state == curr:
                        root = child
                        break
                if root is None:
                    for child in self.root.children:
                        if child is None:
                            continue
                        for grandchild in child.children:
                            if grandchild is None:
                                continue
                            if grandchild.state == curr:
                                root = grandchild 
                                break
                        if root is not None:
                            break
            if root is not None:
                legal_moves = create_mask(root.moves)
                if root is not self.root:
                    self.root = root
                    if training:
                        self.root.policy_evals = (1-NOISE_EPSILON) * self.root.policy_evals + NOISE_EPSILON * self.rng.dirichlet([NOISE_ALPHA] * len(self.root.policy_evals))
        
        # set root if no existing tree is found
        if root is None:
            moves = [encode(m, state.turn) for m in state.get_moves()]
            legal_moves = create_mask(moves)    # to be returned if training

            root_eval = model.forward(X.reshape(1, -1), mask=legal_moves.reshape(1, -1))
            P = root_eval[0][0, moves] if not training else (1-NOISE_EPSILON) * root_eval[0][0, moves] + NOISE_EPSILON * self.rng.dirichlet([NOISE_ALPHA] * len(moves))
            self.root = Node(moves, P, state.state_to_string(), (2*(1-state.turn))-1) 

        # now run sims
        n_sims = num_simulations or NUM_SIMULATIONS
        if total_visits:         # used for UI: budget is the tree's total size, including what was reused
            n_sims = max(0, n_sims - int(self.root.children_visits.sum()))
        for sim in range(n_sims):
            node = self.root
            path = []
            # Loop through until find a leaf node
            cap = self.root.proven_plies - 1 if keep_searching and self.root.proven == 1 else float('inf')  # depth cap (optional)
            while node.terminal is None and (node.proven is None or keep_searching):    # stop at proven nodes if not in keep_searching mode
                if len(path) >= cap:
                    break
                Q = np.divide(node.children_values, node.children_visits, out= np.zeros_like(node.children_values), where= node.children_visits > 0)
                # add moves_left term for leela
                visits = node.children_visits.sum()
                visited = node.children_visits > 0
                if visits > 0:
                    Q_parent = node.children_values.sum() / visits
                    if abs(Q_parent) >= MOVES_LEFT_THRESHOLD:
                        M = np.divide(node.children_moves_left, node.children_visits, out=np.zeros_like(Q), where=visited)
                        M_parent = node.children_moves_left.sum() / visits.sum()
                        move_bonus = -np.sign(Q_parent) * MOVES_LEFT_SCALE * np.clip(M - M_parent, -MOVES_LEFT_CLIP, MOVES_LEFT_CLIP)
                        Q += np.where(visited, move_bonus, 0)      # no bonus for unvisited children

                score = Q + EXPLORATION_CONSTANT * node.policy_evals * np.sqrt(np.sum(node.children_visits)+1) / (node.children_visits+1)
                for i, child in enumerate(node.children):
                    if child is not None and child.proven is not None:
                        if child.proven == 1: 
                            score[i] = -np.inf
                        elif keep_searching is False and child.proven == -1:
                            score[i] = np.inf

                a = np.argmax(score)
                path.append((node, a))
                if node.children[a] is None:
                    break
                node = node.children[a]

            # Expand leaf node
            if not path:             # root is proven (normal mode); nothing to search
                break
            parent, pa = path[-1]
            if node.terminal is not None:
                term = node.terminal * (1 - WIN_DECAY * len(path))
                moves_left = 0.0
            elif node is not parent:            # stopped on an existing node: proven (normal mode) or depth cap
                if node.proven is not None and keep_searching is False:
                    term = node.proven * node.color
                    moves_left = 0.0
                else:                           # depth cap: reuse node's current estimate
                    term = parent.children_values[pa] / parent.children_visits[pa] * parent.color
                    moves_left = max(parent.children_moves_left[pa] / parent.children_visits[pa] - 1 / 160, 0.0)
            else:
                leaf_state = State(node.state)
                term = terminal_value(leaf_state.play_move(decode(node.moves[a], (1-node.color)//2)))  # converts node.color (+1 / -1) to turn (0 for white, 1 for black)
                # regular node
                if term is None:
                    moves = [encode(m, leaf_state.turn) for m in leaf_state.get_moves()]
                    leaf_eval = model.forward(create_board(leaf_state).reshape(1, -1), mask=create_mask(moves).reshape(1, -1))
                    node.children[a] = Node(moves, leaf_eval[0][0, moves], leaf_state.state_to_string(), (2*(1-leaf_state.turn))-1)
                    term = leaf_eval[1][0, 0] * node.children[a].color      # to pull value out of output shape (1, 1)
                    moves_left = leaf_eval[2][0, 0] if len(leaf_eval) > 2 else 0      # for leela
                # game end node
                else:
                    node.children[a] = Node([], np.array([]), leaf_state.state_to_string(), (2*(1-leaf_state.turn))-1, term)
                    moves_left = 0.0
                    term = term * (1 - WIN_DECAY * len(path))
                    if term != 0:
                        node.children[a].proven = -1    # side to move in a position where checkmated has lost
                        node.children[a].proven_plies = 0   # 0 moves from checkmate

            # backprop
            for i, (node, a) in enumerate(path[::-1]):
                node.children_visits[a] += 1
                node.children_values[a] += term * node.color
                node.children_moves_left[a] += moves_left + (i+1) / 160    # node is (i+1)/160 plies away from terminal node that generated moves_left

                child = node.children[a]    # found a winning move
                if child is not None and child.proven == -1:
                    node.proven = 1
                    node.proven_plies = 1 + min(c.proven_plies for c in node.children if c is not None and c.proven == -1)  # ensure not to count children without proven_plies
                elif child is not None and child.proven == 1:
                    if all(c is not None and c.proven == 1 for c in node.children):    # ensure all children are proven mates
                        node.proven = -1
                        node.proven_plies = 1 + max(c.proven_plies for c in node.children)
                if node.proven is not None:
                    term = node.proven * node.color

            # for UI, returns progress in real time
            if progress is not None and (sim == 0 or (sim + 1) % PROGRESS_EVERY == 0):
                if progress(self.root, sim + 1) is False:
                    break
            
            if self.root.proven is not None and keep_searching is False:
                if progress is not None:
                    progress(self.root, sim + 1)    # update UI before breaking
                break

        # After running all simulations, return move
        wins = [i for i, c in enumerate(self.root.children) if c is not None and c.proven == -1]
        if wins:
            a = min(wins, key= lambda i: self.root.children[i].proven_plies)    # take shortest mate move
        elif training:     # temperature to sample moves
            T = max(T_MIN, T_START * 0.5 ** (state.move_number / T_HALFLIFE))  # temperature formula
            v = self.root.children_visits.astype(np.float64)
            v[v < VISIT_CUTOFF * v.max()] = 0          # don't sample clear blunders
            probs = (v / v.max()) ** (1 / T)
            a = self.rng.choice(len(v), p= (probs / probs.sum()))
        else:
            a = np.argmax(self.root.children_visits)
        
        if training:
            pi = np.zeros((BOARD_SIZE**4,))
            pi[self.root.moves] = self.root.children_visits / self.root.children_visits.sum()   # train to estimate MCTS visit counts
            return decode(self.root.moves[a], state.turn), (X, pi, self.root.color, legal_moves, self.root.children_values.sum() / self.root.children_visits.sum())

        return decode(self.root.moves[a], state.turn)    


def calculate_move(state, model, training=False, create_board=State.create_onehot_board, num_simulations=None, progress=None):
    """One-off search with a fresh tree (no reuse).
    Wrapper for scripts that don't need to reuse an MCTS tree."""
    return MCTS().calculate_move(state, model, training, create_board, num_simulations, progress)



