"""
Self-play games run inside worker processes.

Nothing here is tied to a specific agent: the agent class is passed in with each task. It must follow
the MCTSAgent interface: cls(load_checkpoint=False), agent.model.load(path), and
agent.get_move(state, training=True) -> (move, (X, pi, color, mask, q)).

Each worker process keeps one agent per class and reloads its weights from disk whenever the
trainer publishes a new version, so all workers play with the trainer's current network.

Usage (from the trainer):
    with Pool(WORKERS) as pool:
        for X, pi, v, moves_left, mask, q, term, stats in self_play_games(pool, AgentClass, weights_path, version, n_games, seed_base, starts):
            ...
"""

import numpy as np

from Engine.state import State
from Model.util import terminal_value


# per-process cache: agent class -> (weights version, agent instance)
_agents = {}


def _get_agent(agent_cls, weights_path, version):
    """Return this process's agent of agent_cls, loading new weights if the version changed."""
    loaded_version, agent = _agents.get(agent_cls, (None, None))
    if agent is None:
        agent = agent_cls(load_checkpoint=False)
    if loaded_version != version:
        agent.model.load(weights_path)
    _agents[agent_cls] = (version, agent)
    return agent


def play_game(args):
    """
    Play one self-play game. Runs in a worker process.

    args: (agent_cls, weights_path, version, seed, start)
        agent_cls: the agent class to play with (e.g. Agents.AIs.unregularized.Agent)
        weights_path: .npz file with the weights to play with
        version: changes whenever the weights file is rewritten, so workers know to reload
        seed: unused for now (the search's own RNG, agent.mcts.rng, is unseeded); kept so games can be made
            reproducible later by seeding that RNG with it
        start: State string to start from (e.g. a saved endgame position), or None for the normal opening

    Returns (X, pi, v, moves_left, mask, q, term, stats):
        X: (k, input size) network inputs (300 for one-hot boards, 302 for Leela), pi: (k, 625) visit distributions,
        v: (k, 1) outcomes from the side to move's perspective,
        moves_left: (k, 1) plies left in the game from each position, divided by MAX_PLIES (target for a moves-left head),
        mask: (k, 625) legal moves, q: (k, 1) root search Q from the side to move's perspective,
        term: +1/0/-1 from White's view,
        stats: dict summarizing the game (see _game_stats)
    """
    agent_cls, weights_path, version, seed, start = args
    agent = _get_agent(agent_cls, weights_path, version)
    agent.mcts.reset()                      # don't carry the previous game's search tree into this one

    game = State(start) if start else State()
    X, pi, color, mask, q = [], [], [], [], []
    materials, piece_counts = [], []        # per position, for the game stats
    endgame_start = None                    # first endgame position reached, saved as a possible start position
    while True:
        mat, n_pieces = _material(game)
        materials.append(mat)
        piece_counts.append(n_pieces)
        stable = len(materials) > STABLE_PLIES and all(m == mat for m in materials[-STABLE_PLIES - 1:])
        if start is None and endgame_start is None and n_pieces <= START_POS_PIECES and stable:   # only save from normal games
            endgame_start = {"state": game.state_to_string(), "material": mat, "pieces": n_pieces, "ply": game.move_number}

        move, (x, p, c, m, root_q) = agent.get_move(game, training=True)
        X.append(x)
        pi.append(p)
        color.append(c)
        mask.append(m)
        q.append(root_q)

        end = game.play_move(move)
        term = terminal_value(end)
        if term is not None:
            v = (np.array(color, dtype=np.float32) * term).reshape(-1, 1)
            k = len(X)
            moves_left = (np.arange(k, 0, -1, dtype=np.float32) / MAX_PLIES).reshape(-1, 1)   # position i has k - i plies left
            stats = _game_stats(term, end, materials, piece_counts, _material(game))
            stats["endgame_start"] = endgame_start
            stats["start"] = "endgame" if start else "normal"
            return np.stack(X), np.stack(pi), v, moves_left, np.stack(mask), np.array(q, dtype=np.float32).reshape(-1, 1), term, stats


MAX_PLIES = 160         # the 80-move rule: no game lasts longer, so moves_left targets are in (0, 1]

# game stats thresholds
BIG_LEAD = 4            # material lead counted as "clearly winning"
LEAD_WINDOW = 4         # max leads use the material averaged over this many plies, so trades / in-between moves don't count
ENDGAME_PIECES = 4      # positions with <= this many non-king pieces (both sides) count as endgame
START_POS_PIECES = 6    # the first position with <= this many non-king pieces is saved as a possible endgame start
STABLE_PLIES = 2        # ...once material hasn't changed for this many plies, so no trade is half-finished
PIECE_VALUES = {"q": 9, "r": 5, "b": 3, "n": 3, "p": 1}
END_REASONS = ["checkmate", "stalemate", "threefold", "80-move", "insufficient"]   # matched against State's terminal message


def _material(state):
    """Return (material balance from White's view, number of non-king pieces on the board)."""
    mat, n_pieces = 0, 0
    for piece, value in PIECE_VALUES.items():
        w, b = len(state.pieces[("w", piece)]), len(state.pieces[("b", piece)])
        mat += value * (w - b)
        n_pieces += w + b
    return mat, n_pieces


def _end_reason(end):
    """Short label for how the game ended, from State's terminal message (e.g. "Draw by stalemate.")."""
    for reason in END_REASONS:
        if reason in end:
            return reason
    return end      # unknown message: keep it as is so it shows up in the log


def _game_stats(term, end, materials, piece_counts, final):
    """
    Summary of one game, one row of the game log. Material is from White's view.
        result: +1/0/-1 from White's view
        end_reason: checkmate / stalemate / threefold / 80-move / insufficient
        plies: game length
        max_white_lead / max_black_lead: biggest lead each side reached (>= 0), using the material
            averaged over LEAD_WINDOW consecutive plies
        first_big_lead_ply: first ply where either side's averaged lead (as above) is >= BIG_LEAD,
            i.e. the start of the first such window (-1 if never)
        big_lead_side: +1 White / -1 Black / 0 none, the side that got that first big lead
        endgame_plies: positions with <= ENDGAME_PIECES non-king pieces
        final_material / final_pieces: material balance and non-king piece count at the end
        material: material balance at every position, index = ply (the full trajectory, for plots)
    play_game also adds:
        start: "normal" (opening) or "endgame" (started from a saved endgame position)
        endgame_start: None if the game started from an endgame or never reached a position with <= START_POS_PIECES
            non-king pieces and material unchanged for the last STABLE_PLIES plies, otherwise
            {"state": State string, "material": balance from White's view, "pieces": non-king pieces, "ply": move number}
            for the first such position (the side to move there is in the state string)
    """
    mats = np.array(materials)
    traj = np.append(mats, final[0])          # include the final position, its material is settled
    window = min(LEAD_WINDOW, len(traj))
    smoothed = np.convolve(traj, np.ones(window) / window, mode="valid")   # rolling mean
    big = np.flatnonzero(np.abs(smoothed) >= BIG_LEAD)    # smoothed[i] averages plies i .. i+window-1
    first_big = int(big[0]) if len(big) else -1
    return {
        "result": term,
        "end_reason": _end_reason(end),
        "plies": len(materials),
        "max_white_lead": round(float(max(smoothed.max(), 0)), 2),
        "max_black_lead": round(float(max(-smoothed.min(), 0)), 2),
        "first_big_lead_ply": first_big,
        "big_lead_side": int(np.sign(smoothed[first_big])) if first_big >= 0 else 0,
        "endgame_plies": int(np.sum(np.array(piece_counts) <= ENDGAME_PIECES)),
        "final_material": final[0],
        "final_pieces": final[1],
        "material": [int(m) for m in materials] + [int(final[0])],
    }


def self_play_games(pool, agent_cls, weights_path, version, n_games, seed_base, starts=None):
    """
    Play n_games in parallel and yield each game's result as it finishes (in any order).
    seed_base should differ every cycle, e.g. cycle * 100_000, so no two games share a seed.
    starts: optional list of n_games start positions (State strings, or None for the normal opening).
    """
    starts = starts or [None] * n_games
    tasks = [(agent_cls, weights_path, version, seed_base + g, starts[g]) for g in range(n_games)]
    yield from pool.imap_unordered(play_game, tasks)
