"""
Play evaluation games between a model agent and an opponent agent, in parallel worker processes.

Nothing here is tied to a specific agent: the agent classes are passed in with each task.
Model agents must follow the Unregularized interface:
    cls(load_checkpoint=False), agent.model.load(path), agent.get_move(state)
Opponents (Greedy, Rand, ...) only need cls() and get_move(state).

Each worker keeps one instance per agent class and reloads weights only when it is handed a
different checkpoint path (same pattern as Training/self_play.py).

Games start from balanced openings (Eval/opening_book.py): random openings that the checkpoint being
evaluated scores within +-MAX_Q, chosen again at every eval. Each opening is played twice, the model taking
White in one game and Black in the other, so whatever advantage an opening still gives cancels out within the pair.

Usage:
    with Pool(WORKERS) as pool:
        w, d, l = eval_games(pool, Unregularized, "path/to/gen_100.npz", Greedy, n_games=50, openings_path="gen_100.json")
"""

import csv
import re
from pathlib import Path

from Eval.opening_book import opening_state, save_book, select_openings
from Model.util import terminal_value


# per-process cache: agent class -> (checkpoint path or None, agent instance)
_agents = {}


def _get_agent(cls, path=None):
    """Return this process's instance of cls, loading weights from path if it changed."""
    cached = _agents.get(cls)
    if cached is None:
        agent = cls(load_checkpoint=False) if path is not None else cls()
        cached = (None, agent)
    loaded_path, agent = cached
    if path is not None and loaded_path != path:
        agent.model.load(path)
    _agents[cls] = (path, agent)
    return agent


def play_eval_game(args):
    """
    Play one game of a model agent against an opponent. Runs in a worker process.

    args: (model_cls, checkpoint_path, opp_cls, opening_moves, model_white)
        opening_moves: a book opening's moves, played from the start position before the agents take over
    Returns +1 / 0 / -1 from the model's perspective.
    """
    model_cls, path, opp_cls, opening_moves, model_white = args
    model = _get_agent(model_cls, path)
    model.mcts.reset()                    # don't reuse the previous game's tree (e.g. the same opening, other color)
    opp = _get_agent(opp_cls)
    state = opening_state(opening_moves)
    model_turn = 0 if model_white else 1

    while True:
        if state.turn == model_turn:
            move = model.get_move(state)      # no training flag: no noise, no sampling
        else:
            move = opp.get_move(state)
        term = state.play_move(move)
        if term is not None:
            result = terminal_value(term)     # +1 White wins, -1 Black wins, 0 draw
            return result if model_white else -result


def eval_games(pool, model_cls, path, opp_cls, n_games, openings_path=None):
    """
    Play n_games of model (weights from path) vs opponent on an existing Pool.
    First picks n_games // 2 openings the model scores as balanced (opening_book.select_openings), then plays
    each twice: once with the model as White, once as Black. n_games must be even.
    openings_path: if given, the chosen openings are saved there (a book file the UI can load with --book).
    Returns (wins, draws, losses) from the model's perspective.
    """
    if n_games % 2:
        raise ValueError(f"n_games={n_games}: needs an even number of games (each opening is played from both sides)")
    book = select_openings(pool, model_cls, path, n_games // 2)
    if openings_path is not None:
        save_book(book, openings_path)
    tasks = [(model_cls, str(path), opp_cls, opening["moves"], model_white)
             for opening in book["openings"] for model_white in (True, False)]
    results = pool.map(play_eval_game, tasks)
    return results.count(1), results.count(0), results.count(-1)


def list_checkpoints(checkpoint_dir, every=1):
    """[(cycle, path)] for gen_*.npz in checkpoint_dir whose cycle is a multiple of every, sorted."""
    out = []
    for p in Path(checkpoint_dir).glob("gen_*.npz"):
        m = re.fullmatch(r"gen_(\d+)\.npz", p.name)
        if m and int(m.group(1)) % every == 0:
            out.append((int(m.group(1)), p))
    return sorted(out)


CSV_HEADER = ["cycle", "wins", "draws", "losses", "games"]


def done_cycles(csv_path):
    """Cycles already written to the CSV (empty set if it doesn't exist yet)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {int(row["cycle"]) for row in csv.DictReader(f)}


def append_row(csv_path, cycle, wins, draws, losses):
    """Append one result row, creating the file (and its folder) with a header if needed."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(CSV_HEADER)
        writer.writerow([cycle, wins, draws, losses, wins + draws + losses])
