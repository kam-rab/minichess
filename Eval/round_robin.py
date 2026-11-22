"""
Round robin between any set of agents / checkpoints, on the fixed eval positions (Eval/eval_positions/).

Every pair plays every position twice (each side once), with the same games as the eval suite (Eval/eval_suite.py:
no noise, most-visited move, the agents' own sim counts). Games are deterministic, so a pair never needs replaying:
pairs already in the CSV are skipped and only new pairs are played. Adding a model to a finished tournament plays
just its games.

Models (--models), each one of:
    greedy, random              agents without weights
    gen_8000  or  8000          a training checkpoint:   Training/Checkpoints/<agent>/gen_8000.npz
    avg16_8000                  an averaged checkpoint:  Training/Checkpoints/<agent>_avg/avg16_8000.npz
    release                     the agent's RELEASE_WEIGHTS
    path/to/weights.npz         any weights file
    l2:gen_500                  prefix with another agent (a module in Agents/AIs) to mix agents; default --agent

Results: one row per game in Eval/tournaments/<name>.csv. After playing, prints the crosstable (row's score against
column) and ratings: a Bradley-Terry fit on all games between the listed models (draws as half a point), in Elo,
with the first listed model at 0, +- one standard error relative to it.

Usage (from the repo root):
    python -m Eval.round_robin --models avg16_6500 avg16_7000 avg16_7500 avg16_8000
    python -m Eval.round_robin --name avgs --models avg16_6500 avg16_8000 gen_8000 greedy --positions 50
    python -m Eval.round_robin --name avgs --models avg16_6500 avg16_8000 gen_8000 greedy --report   # no new games
"""

import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")    # one thread per worker process
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import importlib
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from Eval.eval_suite import default_positions, play_matchup, subset_positions
from Eval.opening_book import load_book

RESULTS_DIR = Path(__file__).resolve().parent / "tournaments"
FIELDS = ["a", "b", "position", "a_white", "result", "plies"]     # result: +1 / 0 / -1 from a's view
WORKERS = 7
ELO = 400 / np.log(10)          # Bradley-Terry rating units -> Elo
PRIOR_SD = 5.0                  # weak prior on ratings (BT units, ~870 Elo) so a 100% score doesn't diverge
NO_WEIGHTS = {"greedy": "Agents.greedy_material", "random": "Agents.rand"}


def resolve(spec, default_agent):
    """'[agent:]name' -> (label, agent class, weights path or None)."""
    if spec in NO_WEIGHTS:
        return spec, importlib.import_module(NO_WEIGHTS[spec]).Agent, None
    agent, _, name = spec.rpartition(":")
    agent = agent or default_agent
    cls = importlib.import_module(f"Agents.AIs.{agent}").Agent
    name = f"gen_{name}" if name.isdigit() else name
    label = name if agent == default_agent else f"{agent}:{name}"
    if name.endswith(".npz"):
        path = Path(name)
    elif name == "release":
        path = cls.RELEASE_WEIGHTS
    else:
        ckpt_dir = cls.CHECKPOINT_DIR
        path = next((p for p in (ckpt_dir / f"{name}.npz", ckpt_dir.parent / f"{ckpt_dir.name}_avg" / f"{name}.npz")
                     if p.exists()), ckpt_dir / f"{name}.npz")
    if not path.exists():
        raise FileNotFoundError(f"{spec}: no weights at {path}")
    return label, cls, path


def play_pair(pool, a, b, positions, indices):
    """All games between models a and b (label, class, path). Returns rows with the result from a's view."""
    flip = a[2] is None                   # play_matchup needs the first model to have weights
    first, second = (b, a) if flip else (a, b)
    rows = play_matchup(pool, first[1], first[2], second[1], second[2], positions, indices)
    return [{"a": a[0], "b": b[0], "position": r["position"],
             "a_white": int(r["model_white"]) ^ int(flip), "result": -r["result"] if flip else r["result"],
             "plies": r["plies"]} for r in rows]


def load_games(csv_path):
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def append_games(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def tally(games, labels):
    """points[i][j], n[i][j]: i's points and games against j, over games between the listed labels."""
    idx = {l: i for i, l in enumerate(labels)}
    points, n = np.zeros((len(labels),) * 2), np.zeros((len(labels),) * 2)
    for g in games:
        if g["a"] in idx and g["b"] in idx:
            i, j, s = idx[g["a"]], idx[g["b"]], (int(g["result"]) + 1) / 2
            points[i, j] += s
            points[j, i] += 1 - s
            n[i, j] += 1
            n[j, i] += 1
    return points, n


def ratings(points, n):
    """Bradley-Terry ratings in Elo (model 0 fixed at 0) and their standard errors, by Newton's method."""
    k = len(points)
    r = np.zeros(k)
    for _ in range(100):
        p = 1 / (1 + np.exp(-(r[:, None] - r[None, :])))          # p[i, j]: expected score of i against j
        grad = (points - n * p).sum(axis=1) - r / PRIOR_SD ** 2
        w = n * p * (1 - p)
        hess = w - np.diag(w.sum(axis=1)) - np.eye(k) / PRIOR_SD ** 2
        step = np.linalg.solve(hess[1:, 1:], grad[1:])           # model 0 stays at 0
        r[1:] -= step
        if np.abs(step).max() < 1e-9:
            break
    se = np.sqrt(np.clip(np.diag(np.linalg.inv(-hess[1:, 1:])), 0, None))
    return r * ELO, np.concatenate([[0.0], se * ELO])


def report(games, labels):
    points, n = tally(games, labels)
    width = max(10, max(map(len, labels)) + 2)
    print("\nscore of row against column (games per pair: %s)" % ", ".join(sorted({str(int(x)) for x in n[n > 0]})))
    print(" " * width + "".join(f"{l[:width - 2]:>{width}}" for l in labels) + f"{'total':>{width}}")
    for i, l in enumerate(labels):
        cells = ["-" if i == j or n[i, j] == 0 else f"{points[i, j] / n[i, j]:.3f}" for j in range(len(labels))]
        total = points[i].sum() / n[i].sum() if n[i].sum() else float("nan")
        print(f"{l:<{width}}" + "".join(f"{c:>{width}}" for c in cells) + f"{total:>{width}.3f}")
    if (n.sum(axis=1) == 0).any():
        print("(some models have no games yet; ratings skipped)")
        return
    elo, se = ratings(points, n)
    print(f"\nratings (Elo, {labels[0]} = 0, +- 1 SE relative to it)")
    for i in np.argsort(-elo):
        print(f"  {labels[i]:<{width}} {elo[i]:+7.0f}  +- {se[i]:.0f}")


def main():
    parser = argparse.ArgumentParser(description="Round robin between agents / checkpoints on the eval positions.")
    parser.add_argument("--models", nargs="+", required=True, help="see the module docstring for the forms")
    parser.add_argument("--agent", default="leela", help="agent for models given without a prefix")
    parser.add_argument("--name", default="round_robin", help="tournament name: results in Eval/tournaments/<name>.csv")
    parser.add_argument("--positions", type=int, default=None, help="use a same-mix subset of this many positions")
    parser.add_argument("--book", default=None, help="positions file (default: the agent's eval positions)")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--report", action="store_true", help="only print the table from the CSV, play nothing")
    args = parser.parse_args()

    models = [resolve(s, args.agent) for s in args.models]
    labels = [m[0] for m in models]
    csv_path = RESULTS_DIR / f"{args.name}.csv"
    games = load_games(csv_path)

    if not args.report:
        from multiprocessing import Pool
        default_cls = importlib.import_module(f"Agents.AIs.{args.agent}").Agent
        book = load_book(args.book or default_positions(default_cls))
        positions, indices = book["openings"], subset_positions(book, args.positions)
        played = {frozenset((g["a"], g["b"])) for g in games}
        todo = [(a, b) for a, b in combinations(models, 2) if frozenset((a[0], b[0])) not in played]
        print(f"{len(todo)} pairs to play ({len(models) * (len(models) - 1) // 2 - len(todo)} already in {csv_path.name}), "
              f"{2 * len(indices)} games each")
        with Pool(args.workers) as pool:
            for a, b in todo:
                start = time.time()
                rows = play_pair(pool, a, b, positions, indices)
                append_games(csv_path, rows)          # whole pair at once, so a stopped run never leaves half of one
                games += rows
                s = np.mean([(r["result"] + 1) / 2 for r in rows])
                print(f"  {a[0]} vs {b[0]}: {s:.3f}   ({time.time() - start:.0f}s)", flush=True)
    report(games, labels)


if __name__ == "__main__":
    main()
