"""
Build a fixed, Q-stratified set of eval positions, judged by one frozen checkpoint.

Two sources of positions:
    random:   N_RANDOM positions after 4-6 uniformly random plies from the start
    selfplay: N_SELFPLAY positions from the frozen model's own self-play (training settings: noise + temperature),
              a third each taken at plies 4-6, 7-12 and 13-18 (one position per game, so they're independent)

Every candidate is scored by the frozen model's search (Q of the most visited move, White's view, SIMS sims),
and kept only while its |Q| bin still needs positions. Within each source (and each self-play ply range):
    40% |Q| <= 0.2        (balanced)
    40% 0.2 < |Q| <= 0.5  (one side better)
    20% |Q| > 0.5         (one side clearly winning)
Each position is played twice in an eval, the evaluated model taking each side once, so seen from the model's side
this gives 40% balanced, 20% / 20% better / worse, 10% / 10% clearly winning / losing.

The positions are saved as a book (same format as Eval/opening_book.py), so the UI can show them:
    python -m UI --book Eval/eval_positions/leela_gen_1000.json

Generate (from the repo root):
    python -m Eval.eval_positions --agent leela --checkpoint gen_1000
    python -m Eval.eval_positions --agent leela --checkpoint gen_1000 --sims 20000 --random 25 --selfplay 50 --out path.json

Everything is seeded, so the same arguments give the same positions (given the same code and weights).
"""

import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")    # one thread per worker process
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import importlib
import random
from pathlib import Path

import numpy as np

from Engine.state import State
from Eval.opening_book import save_book


OUT_DIR = Path(__file__).resolve().parent / "eval_positions"

N_RANDOM = 25                     # random-opening positions (each played twice -> 50 games)
N_SELFPLAY = 50                   # self-play positions (each played twice -> 100 games)
RANDOM_PLIES = (4, 6)             # inclusive ply range for random openings
SELFPLAY_PLIES = [(4, 6), (7, 12), (13, 18)]    # self-play ply ranges, split evenly
Q_BINS = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.01)]  # |Q| bins: (low, high], the first includes 0
Q_SHARES = [0.4, 0.4, 0.2]        # target share of each bin
BIN_NAMES = ["balanced", "better", "winning"]
SIMS = 20000                      # search size for scoring each candidate
PER_ROUND = 12                    # candidates generated per unfilled stratum per round
MAX_CANDIDATES = 4000             # per stratum; error if the quotas still aren't met
WORKERS = 6


# ---------------------------------------------------------------- quotas

def _quotas(sizes, shares):
    """Integer counts[stratum][bin] whose rows sum to sizes and whose columns are as close as possible to
    sum(sizes) * shares: largest-remainder rounding over the whole table."""
    totals = _round_to(sum(sizes), shares)
    exact = np.outer(sizes, shares)
    counts = np.floor(exact).astype(int)
    rem = exact - counts
    for i, j in sorted(np.ndindex(counts.shape), key=lambda ij: -rem[ij]):
        if counts[i].sum() < sizes[i] and counts[:, j].sum() < totals[j]:
            counts[i, j] += 1
    return counts.tolist()


def _round_to(n, shares):
    exact = np.array(shares) * n
    counts = np.floor(exact).astype(int)
    for j in np.argsort(-(exact - counts))[: n - counts.sum()]:
        counts[j] += 1
    return counts


def _bin(q):
    a = abs(q)
    return next(i for i, (lo, hi) in enumerate(Q_BINS) if (a <= hi and (a > lo or i == 0)))


# ---------------------------------------------------------------- candidates (run in worker processes)

_agent = None


def _init_worker(agent_module, checkpoint_path):
    global _agent
    _agent = importlib.import_module(agent_module).Agent(load_checkpoint=False)
    _agent.model.load(checkpoint_path)


def _random_position(seed, lo, hi):
    """Uniformly random moves for a seeded number of plies in [lo, hi]. None if the game ends first."""
    rng = random.Random(seed)
    plies = rng.randint(lo, hi)
    state, moves = State(), []
    for _ in range(plies):
        move = rng.choice(state.get_moves())
        moves.append(move)
        if state.play_move(move):
            return None
    return state, moves


def _selfplay_position(seed, lo, hi):
    """The frozen model's self-play (training settings), stopped at a seeded ply in [lo, hi]. None if the game ends first."""
    plies = random.Random(seed).randint(lo, hi)
    _agent.mcts.reset()
    _agent.mcts.rng = np.random.default_rng(seed)       # noise and temperature sampling, reproducible
    state, moves = State(), []
    for _ in range(plies):
        move, _ = _agent.get_move(state, training=True)
        move = tuple(map(tuple, move))
        moves.append(move)
        if state.play_move(move):
            return None
    return state, moves


def _score(state, sims):
    """Q of the most visited move after a search, White's view (+-1 if a result is proven)."""
    _agent.mcts.reset()
    _agent.mcts.calculate_move(state, _agent.model, False, _agent.CREATE_BOARD, num_simulations=sims)
    root = _agent.mcts.root
    if getattr(root, "proven", None) is not None:
        return float(root.proven * root.color)
    best = int(np.argmax(root.children_visits))
    return float(root.children_values[best] / root.children_visits[best] * root.color)


def _candidate(args):
    """Make and score one candidate. Returns a position dict, or None if the game ended before the target ply."""
    source, lo, hi, seed, sims = args
    made = _random_position(seed, lo, hi) if source == "random" else _selfplay_position(seed, lo, hi)
    if made is None:
        return None
    state, moves = made
    q = _score(state, sims)
    return {"source": source, "seed": seed, "ply": len(moves), "moves": moves, "q": round(q, 3), "bin": BIN_NAMES[_bin(q)],
            "state": state.state_to_string()}


# ---------------------------------------------------------------- driver

def generate(agent_name, checkpoint, n_random=N_RANDOM, n_selfplay=N_SELFPLAY, sims=SIMS, out=None):
    from multiprocessing import Pool

    agent_module = f"Agents.AIs.{agent_name}"
    agent_cls = importlib.import_module(agent_module).Agent
    ckpt = agent_cls.CHECKPOINT_DIR / (checkpoint if checkpoint.endswith(".npz") else checkpoint + ".npz")
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    out = Path(out) if out else OUT_DIR / f"{agent_name}_{ckpt.stem}.json"

    # strata: (source, ply range, size), each with its own |Q| bin quotas
    strata = [("random", RANDOM_PLIES)]
    sizes = [n_random]
    sp_sizes = _round_to(n_selfplay, [1 / len(SELFPLAY_PLIES)] * len(SELFPLAY_PLIES))
    strata += [("selfplay", r) for r in SELFPLAY_PLIES]
    sizes += [int(s) for s in sp_sizes]
    quotas = [_quotas([sizes[0]], Q_SHARES)[0]] + _quotas(sizes[1:], Q_SHARES)

    kept = [[[] for _ in Q_BINS] for _ in strata]
    seen = set()
    next_seed = [k * 1_000_000 for k in range(len(strata))]      # disjoint seed ranges per stratum
    tried = [0] * len(strata)

    def need(k):
        return any(len(kept[k][b]) < quotas[k][b] for b in range(len(Q_BINS)))

    with Pool(WORKERS, initializer=_init_worker, initargs=(agent_module, str(ckpt))) as pool:
        while any(need(k) for k in range(len(strata))):
            tasks = []
            for k, (source, (lo, hi)) in enumerate(strata):
                if not need(k):
                    continue
                if tried[k] >= MAX_CANDIDATES:
                    raise RuntimeError(f"{source} plies {lo}-{hi}: quotas {quotas[k]} not met after {tried[k]} candidates "
                                       f"(have {[len(b) for b in kept[k]]})")
                tasks += [(k, (source, lo, hi, next_seed[k] + i, sims)) for i in range(PER_ROUND)]
                next_seed[k] += PER_ROUND
                tried[k] += PER_ROUND
            results = pool.map(_candidate, [t for _, t in tasks])
            for (k, _), pos in zip(tasks, results):        # in seed order within each stratum
                if pos is None or pos["state"] in seen:
                    continue
                b = BIN_NAMES.index(pos["bin"])
                if len(kept[k][b]) < quotas[k][b]:
                    kept[k][b].append(pos)
                    seen.add(pos["state"])
            for k, (source, (lo, hi)) in enumerate(strata):
                print(f"{source:8s} plies {lo:2d}-{hi:2d}: {[len(b) for b in kept[k]]} of {quotas[k]}   ({tried[k]} tried)")
            print()

    positions = [p for k in range(len(strata)) for b in range(len(Q_BINS)) for p in kept[k][b]]
    book = {"agent": agent_name, "checkpoint": ckpt.name, "sims": sims, "q_bins": Q_BINS, "q_shares": Q_SHARES,
            "random_plies": RANDOM_PLIES, "selfplay_plies": SELFPLAY_PLIES, "candidates_tried": sum(tried),
            "openings": positions}       # "openings" so load_book and the UI can read it
    save_book(book, out)
    print(f"saved {len(positions)} positions to {out}")
    return book


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Q-stratified eval position set.")
    parser.add_argument("--agent", required=True, help="module in Agents/AIs, e.g. leela")
    parser.add_argument("--checkpoint", required=True, help="frozen checkpoint in that agent's CHECKPOINT_DIR, e.g. gen_1000")
    parser.add_argument("--random", type=int, default=N_RANDOM, help="random-opening positions")
    parser.add_argument("--selfplay", type=int, default=N_SELFPLAY, help="self-play positions")
    parser.add_argument("--sims", type=int, default=SIMS)
    parser.add_argument("--out", default=None, help=f"default: {OUT_DIR}/<agent>_<checkpoint>.json")
    args = parser.parse_args()
    generate(args.agent, args.checkpoint, args.random, args.selfplay, args.sims, args.out)
