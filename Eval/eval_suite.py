"""
Evaluate checkpoints on the fixed, Q-stratified eval positions (Eval/eval_positions.py).

Every position is played twice, the evaluated checkpoint taking White once and Black once, against:
    greedy   the greedy material agent, every greedy_every cycles
    fixed    a fixed checkpoint of the same agent (gen_{fixed_gen}, e.g. the frozen model that scored the positions),
             every fixed_every cycles
    moving   the same agent's checkpoint moving_gap cycles earlier, every moving_every cycles

Each opponent can use a subset of the positions (greedy_positions, fixed_positions, moving_positions). A subset
keeps the set's mix: the same share of each source / ply range / |Q| bin (subset_positions), and it's always the
same positions for the same size, so results stay comparable across cycles.

Each game is one row of the results CSV, so results can be grouped any way later. The bucket is the positions'
Q (from the model that scored them) seen from the evaluated model's side (q_model = q_white, negated when the
model plays Black):
    losing  q_model < -0.5      worse  -0.5 <= q_model < -0.2      balanced  |q_model| <= 0.2
    better  0.2 < q_model <= 0.5     winning  q_model > 0.5
expected = (q_model + 1) / 2 is the score that Q predicts, to compare against the actual score.

From the training loop (on its existing Pool), after saving gen_{cycle}:
    eval_checkpoint(pool, agent_class, cycle, greedy_positions=25, fixed_positions=50, moving_positions=50)

Standalone, over saved checkpoints (from the repo root):
    python -m Eval.eval_suite --agent leela
    python -m Eval.eval_suite --agent leela --from 50 --to 1000 --greedy-every 50 --fixed-gen 1000 --moving-gap 100
Only the judging, over saved checkpoints (the matchups turned off):
    python -m Eval.eval_suite --agent leela --greedy-every 0 --fixed-every 0 --moving-every 0
Summarize what's in the CSV (score and score - expected per bucket):
    python -m Eval.eval_suite --agent leela --summary

Matchups already in the CSV are skipped, so reruns (or a resumed training run) never play one twice.

Judging (every judge_every cycles): the checkpoint also scores every eval position itself, with the same search
the positions were scored with (the book's sims), so its view can be followed over training. One row per position
in <agent>_judge_<scoring checkpoint>.csv:
    q_ref   the scoring model's Q (from the book), White's view
    q       this checkpoint's search Q (most visited move, as the book), White's view
    v_net   this checkpoint's raw value head on the position (no search), White's view
At the scoring checkpoint itself, q == q_ref (same search, no noise). Judgings already in the CSV are skipped too.
"""

import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")    # one thread per worker process
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import importlib
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

from Eval.opening_book import load_book, opening_state
from Eval.tourney import list_checkpoints
from Model.util import terminal_value


# defaults for eval_checkpoint
GREEDY_EVERY = 50
FIXED_GEN = 1000
FIXED_EVERY = 100
MOVING_GAP = 100        # must be a multiple of 25: the training loop only keeps every 25th checkpoint
MOVING_EVERY = 100
GREEDY_POSITIONS = 25   # positions per opponent (each played from both sides); None = all of them
FIXED_POSITIONS = 50
MOVING_POSITIONS = 50
JUDGE_EVERY = 50        # the checkpoint scores every position itself (0 / None turns it off)
JUDGE_SIMS = None       # None = the sims the positions were scored with, so q is comparable to q_ref
TEST_EVERY = 50         # print the search's Q of the best / blunder moves in the test positions (0 / None turns it off)
TEST_SIMS = 3000

WORKERS = 6             # standalone runs only; the training loop passes its own Pool
POSITIONS_DIR = Path(__file__).resolve().parent / "eval_positions"
RESULTS_DIR = Path(__file__).resolve().parent / "training"
TEST_POSITIONS = POSITIONS_DIR / "test_positions.json"     # hand-picked positions the engine got wrong (best vs blunder)

OPPONENTS = ["greedy", "fixed", "moving"]
BUCKETS = ["losing", "worse", "balanced", "better", "winning"]
FIELDS = ["cycle", "opponent", "opponent_checkpoint", "position", "source", "ply", "q_white", "model_white",
          "q_model", "bucket", "expected", "result", "plies"]
Q_BIN_NAMES = ["balanced", "better", "winning"]     # |Q| bins of the saved positions (Eval/eval_positions.py)
JUDGE_FIELDS = ["cycle", "position", "source", "ply", "q_ref", "q", "v_net", "sims"]


def bucket(q_model):
    if q_model < -0.5:
        return "losing"
    if q_model < -0.2:
        return "worse"
    if q_model <= 0.2:
        return "balanced"
    if q_model <= 0.5:
        return "better"
    return "winning"


def default_positions(agent_cls, fixed_gen=FIXED_GEN):
    """Eval/eval_positions/<agent>_gen_<fixed_gen>.json: the positions scored by the fixed model."""
    return POSITIONS_DIR / f"{agent_cls.CHECKPOINT_DIR.name}_gen_{fixed_gen}.json"


def default_csv(agent_cls, positions_path):
    """Eval/training/<agent>/<agent>_suite_<scoring checkpoint>.csv: one results file per position set."""
    name = agent_cls.CHECKPOINT_DIR.name
    return RESULTS_DIR / name / f"{name}_suite_{Path(load_book(positions_path)['checkpoint']).stem}.csv"


def default_judge_csv(agent_cls, positions_path):
    """Eval/training/<agent>/<agent>_judge_<scoring checkpoint>.csv"""
    name = agent_cls.CHECKPOINT_DIR.name
    return RESULTS_DIR / name / f"{name}_judge_{Path(load_book(positions_path)['checkpoint']).stem}.csv"


def subset_positions(book, n):
    """
    Indices of n of the book's positions, keeping its mix: positions are grouped by source, ply range and |Q| bin,
    and each group gives its share of n (largest-remainder rounding). Within a group the first ones are taken, so the
    same n always gives the same positions. n=None or n >= the number of positions gives all of them.
    """
    positions = book["openings"]
    if n is None or n >= len(positions):
        return list(range(len(positions)))
    ranges = book.get("selfplay_plies", [])

    def group(pos):
        ply_range = next((i for i, (lo, hi) in enumerate(ranges) if lo <= pos.get("ply", 0) <= hi), -1) \
            if pos.get("source") == "selfplay" else -1
        return pos.get("source", ""), ply_range, pos.get("bin", bucket_abs(pos["q"]))

    cells = defaultdict(list)                   # (source, ply range) group, |Q| bin -> position indices
    for i, pos in enumerate(positions):
        *g, b = group(pos)
        cells[(tuple(g), b)].append(i)
    group_sizes, bin_sizes = defaultdict(int), defaultdict(int)
    for (g, b), ix in cells.items():
        group_sizes[g] += len(ix)
        bin_sizes[b] += len(ix)
    # targets for each group and each bin, then fill the table toward both (largest remainders first)
    group_take, bin_take = _round_shares(group_sizes, n), _round_shares(bin_sizes, n)
    exact = {c: len(ix) * n / len(positions) for c, ix in cells.items()}
    take = {c: int(exact[c]) for c in cells}
    order = sorted(cells, key=lambda c: (-(exact[c] - take[c]), c))
    for strict in (True, False):                # second pass: only the group targets, in case the bins can't both be met
        for c in order:
            g, b = c
            while (take[c] < len(cells[c]) and sum(t for (g2, _), t in take.items() if g2 == g) < group_take[g]
                   and (not strict or sum(t for (_, b2), t in take.items() if b2 == b) < bin_take[b])):
                take[c] += 1
                if strict:
                    break
    return sorted(i for c, ix in cells.items() for i in ix[:take[c]])


def _round_shares(sizes, n):
    """Split n in proportion to sizes (a dict), largest remainders first."""
    total = sum(sizes.values())
    exact = {k: v * n / total for k, v in sizes.items()}
    out = {k: int(e) for k, e in exact.items()}
    for k in sorted(exact, key=lambda k: (-(exact[k] - out[k]), k))[: n - sum(out.values())]:
        out[k] += 1
    return out


def bucket_abs(q):
    """|Q| bin name, as in Eval/eval_positions.py."""
    return Q_BIN_NAMES[0] if abs(q) <= 0.2 else Q_BIN_NAMES[1] if abs(q) <= 0.5 else Q_BIN_NAMES[2]


def _ckpt(agent_cls, gen):
    return agent_cls.CHECKPOINT_DIR / f"gen_{gen:03d}.npz"


# ---------------------------------------------------------------- games (run in worker processes)

_cache = OrderedDict()      # (class, checkpoint path or None) -> agent; a few at most per process
CACHE_SIZE = 4


def _agent(cls, path):
    key = (cls, path)
    if key not in _cache:
        if path is None:
            agent = cls()                             # e.g. greedy: no weights
        else:
            agent = cls(load_checkpoint=False)
            agent.model.load(path)
        _cache[key] = agent
        if len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    _cache.move_to_end(key)
    return _cache[key]


def _play(args):
    """One game from a position. Returns (task id, result from the model's view, plies played)."""
    task_id, model_cls, model_path, opp_cls, opp_path, moves, model_white = args
    model, opp = _agent(model_cls, model_path), _agent(opp_cls, opp_path)
    for agent in (model, opp):
        if hasattr(agent, "mcts"):
            agent.mcts.reset()                        # no tree carried over from another game
    state = opening_state(moves)
    model_turn = 0 if model_white else 1
    plies = 0
    while True:
        agent = model if state.turn == model_turn else opp
        move = agent.get_move(state)                  # no training flag: no noise, no sampling
        term = state.play_move(move)
        plies += 1
        if term is not None:
            result = terminal_value(term)             # +1 White wins, -1 Black wins, 0 draw
            return task_id, (result if model_white else -result), plies


def _judge(args):
    """Score one position with the model. Returns (task id, search Q, raw value head), both White's view.
    The search Q is taken as in Eval/eval_positions.py: Q of the most visited move, or +-1 if a result is proven."""
    import numpy as np
    from Model.util import create_mask

    task_id, model_cls, model_path, moves, sims = args
    agent = _agent(model_cls, model_path)
    agent.mcts.reset()
    state = opening_state(moves)
    agent.mcts.calculate_move(state, agent.model, False, agent.CREATE_BOARD, num_simulations=sims)
    root = agent.mcts.root
    if getattr(root, "proven", None) is not None:
        q = float(root.proven * root.color)
    else:
        best = int(np.argmax(root.children_visits))
        q = float(root.children_values[best] / root.children_visits[best] * root.color)
    out = agent.model.forward(agent.CREATE_BOARD(state).reshape(1, -1), mask=create_mask(root.moves).reshape(1, -1))
    v_net = float(out[1][0, 0] * root.color)
    agent.mcts.reset()
    return task_id, q, v_net


# ---------------------------------------------------------------- matchups

def play_matchup(pool, model_cls, model_path, opp_cls, opp_path, positions, indices=None):
    """
    Play each position in indices (default: all) twice (model as White, then as Black) on the pool.
    opp_path: the opponent's weights, or None for agents without weights (greedy).
    Returns one dict per game (FIELDS, without cycle / opponent / opponent_checkpoint), ordered by position.
    """
    indices = list(range(len(positions))) if indices is None else indices
    tasks = []
    for i in indices:
        pos = positions[i]
        for model_white in (True, False):
            tasks.append((len(tasks), model_cls, str(model_path), opp_cls, None if opp_path is None else str(opp_path),
                          pos["moves"], model_white))
    rows = [None] * len(tasks)
    for task_id, result, plies in pool.imap_unordered(_play, tasks):
        i, model_white = indices[task_id // 2], tasks[task_id][-1]
        pos = positions[i]
        q_model = pos["q"] if model_white else -pos["q"]
        rows[task_id] = {"position": i, "source": pos.get("source", ""), "ply": pos.get("ply", ""), "q_white": pos["q"],
                         "model_white": int(model_white), "q_model": round(q_model, 3), "bucket": bucket(q_model),
                         "expected": round((q_model + 1) / 2, 3), "result": result, "plies": plies}
    return rows


def due_matchups(agent_cls, cycle, greedy_every=GREEDY_EVERY, fixed_gen=FIXED_GEN, fixed_every=FIXED_EVERY,
                 moving_gap=MOVING_GAP, moving_every=MOVING_EVERY):
    """[(opponent name, opponent class, opponent checkpoint path or None)] due at this cycle.
    An every / gen of None (or 0) turns that opponent off. Missing checkpoints are skipped with a message."""
    from Agents.greedy_material import Agent as Greedy

    due = []
    if cycle == 0:
        return due
    if greedy_every and cycle % greedy_every == 0:
        due.append(("greedy", Greedy, None))
    if fixed_gen is not None and fixed_every and cycle % fixed_every == 0 and cycle != fixed_gen:
        due.append(("fixed", agent_cls, _ckpt(agent_cls, fixed_gen)))
    if moving_gap and moving_every and cycle % moving_every == 0 and cycle - moving_gap >= 0:
        due.append(("moving", agent_cls, _ckpt(agent_cls, cycle - moving_gap)))
    found = [d for d in due if d[2] is None or d[2].exists()]
    for name, _, path in due:
        if path is not None and not path.exists():
            print(f"eval suite: skipping {name} at cycle {cycle}, {path.name} doesn't exist")
    return found


def judge_checkpoint(pool, agent_cls, cycle, positions_path, csv_path, sims=JUDGE_SIMS):
    """
    gen_{cycle} scores every eval position (search Q and raw value head, see the module docstring); the rows are
    appended to csv_path. sims=None uses the book's sims. Skipped if this cycle is already in the CSV.
    Returns the rows written ([] if skipped).
    """
    import numpy as np

    csv_path = Path(csv_path)
    if cycle in _judged(csv_path):
        return []
    book = load_book(positions_path)
    positions = book["openings"]
    sims = sims or book["sims"]
    model_path = str(_ckpt(agent_cls, cycle))
    start = time.time()
    tasks = [(i, agent_cls, model_path, pos["moves"], sims) for i, pos in enumerate(positions)]
    rows = [None] * len(tasks)
    for i, q, v_net in pool.imap_unordered(_judge, tasks):
        pos = positions[i]
        rows[i] = {"cycle": cycle, "position": i, "source": pos.get("source", ""), "ply": pos.get("ply", ""),
                   "q_ref": pos["q"], "q": round(q, 3), "v_net": round(v_net, 3), "sims": sims}
    _append(csv_path, rows, JUDGE_FIELDS)
    q, q_ref = np.array([r["q"] for r in rows]), np.array([r["q_ref"] for r in rows])
    print(f"gen_{cycle:03d} judged {len(rows)} positions: corr with {Path(book['checkpoint']).stem} {np.corrcoef(q, q_ref)[0, 1]:.2f}, "
          f"mean |q - q_ref| {np.mean(np.abs(q - q_ref)):.2f}   ({time.time() - start:.0f}s)", flush=True)
    return rows


def _test(args):
    """Search one test position. Returns (task id, label of the move search picks,
    {label: (Q White's view or None if unvisited, visits)} for the position's best and blunder moves)."""
    import numpy as np
    from Model.util import decode

    task_id, model_cls, model_path, pos, sims = args
    agent = _agent(model_cls, model_path)
    agent.mcts.reset()
    state = opening_state(pos["moves"])
    picked = agent.mcts.calculate_move(state, agent.model, False, agent.CREATE_BOARD, num_simulations=sims)
    root = agent.mcts.root
    found = {_as_tuple(decode(m, state.turn)): i for i, m in enumerate(root.moves)}
    stats = {}
    for key in ("best", "blunder"):
        i = found[_as_tuple(pos[key + "_move"])]
        n = int(root.children_visits[i])
        child = root.children[i]
        if child is not None and child.proven is not None:
            q = float(child.proven * child.color)               # proven is for the side to move at the child
        else:
            q = float(root.children_values[i] / n * root.color) if n else None
        stats[pos[key]] = (q, n)
    labels = {_as_tuple(pos[k + "_move"]): pos[k] for k in ("best", "blunder")}
    agent.mcts.reset()
    return task_id, labels.get(_as_tuple(picked), "other"), stats


def _as_tuple(move):
    return tuple(tuple(int(x) for x in sq) for sq in move)


def test_checkpoint(pool, agent_cls, cycle, positions_path=TEST_POSITIONS, sims=TEST_SIMS):
    """Print, for each test position, the move gen_{cycle} picks and the search's Q (White's view) and visits
    of the position's best and blunder moves. Printout only; nothing is saved."""
    positions_path = Path(positions_path)
    if not positions_path.exists():
        return
    positions = load_book(positions_path)["openings"]
    model_path = str(_ckpt(agent_cls, cycle))
    tasks = [(i, agent_cls, model_path, pos, sims) for i, pos in enumerate(positions)]
    for i, picked, stats in sorted(pool.imap_unordered(_test, tasks), key=lambda r: r[0]):
        pos = positions[i]
        moves = "   ".join(f"{label} q {'-' if q is None else f'{q:+.2f}'} (n {n})" for label, (q, n) in stats.items())
        mark = "ok" if picked == pos["best"] else "MISS"
        print(f"gen_{cycle:03d} test {pos.get('name', i)}: picks {picked} [{mark}]   {moves}   ({sims} sims)", flush=True)


def eval_checkpoint(pool, agent_cls, cycle, positions_path=None, csv_path=None,
                    greedy_every=GREEDY_EVERY, fixed_gen=FIXED_GEN, fixed_every=FIXED_EVERY,
                    moving_gap=MOVING_GAP, moving_every=MOVING_EVERY,
                    greedy_positions=GREEDY_POSITIONS, fixed_positions=FIXED_POSITIONS, moving_positions=MOVING_POSITIONS,
                    judge_every=JUDGE_EVERY, judge_sims=JUDGE_SIMS, judge_csv_path=None,
                    test_every=TEST_EVERY, test_sims=TEST_SIMS):
    """
    Play whichever matchups are due for gen_{cycle} of agent_cls (see due_matchups), append each to the CSV
    and print a one-line score for it. Matchups already in the CSV are skipped.
    *_positions: how many positions each opponent plays (each from both sides, so twice as many games), a subset
    with the same mix as the whole set (subset_positions); None = all.
    positions_path: eval positions file (default: default_positions(agent_cls, fixed_gen)).
    csv_path: results file (default: default_csv(agent_cls, positions_path)).
    judge_every: every this many cycles the checkpoint also scores all the positions itself (judge_checkpoint),
    with judge_sims (None = the book's), into judge_csv_path (default: default_judge_csv). 0 / None turns it off.
    test_every: every this many cycles, print how the checkpoint searches the test positions (test_checkpoint),
    with test_sims. 0 / None turns it off.
    Returns {opponent name: (wins, draws, losses)} for the matchups played.
    """
    model_path = _ckpt(agent_cls, cycle)
    due = due_matchups(agent_cls, cycle, greedy_every, fixed_gen, fixed_every, moving_gap, moving_every)
    judge_due = bool(judge_every) and cycle > 0 and cycle % judge_every == 0
    if test_every and cycle > 0 and cycle % test_every == 0:
        test_checkpoint(pool, agent_cls, cycle, sims=test_sims)
    if not due and not judge_due:
        return {}
    positions_path = positions_path or default_positions(agent_cls, fixed_gen if fixed_gen is not None else FIXED_GEN)
    if judge_due:
        judge_checkpoint(pool, agent_cls, cycle, positions_path,
                         judge_csv_path or default_judge_csv(agent_cls, positions_path), judge_sims)
    if not due:
        return {}
    csv_path = Path(csv_path) if csv_path else default_csv(agent_cls, positions_path)
    book = load_book(positions_path)
    positions = book["openings"]
    n_positions = {"greedy": greedy_positions, "fixed": fixed_positions, "moving": moving_positions}
    done = _done(csv_path)

    results = {}
    for opp_name, opp_cls, opp_path in due:
        if (cycle, opp_name) in done:
            continue
        start = time.time()
        rows = play_matchup(pool, agent_cls, model_path, opp_cls, opp_path, positions,
                            subset_positions(book, n_positions[opp_name]))
        for r in rows:
            r.update(cycle=cycle, opponent=opp_name, opponent_checkpoint="" if opp_path is None else Path(opp_path).stem)
        _append(csv_path, rows)                      # whole matchup at once, so a stopped run never leaves half of one
        w = sum(r["result"] == 1 for r in rows)
        d = sum(r["result"] == 0 for r in rows)
        l = len(rows) - w - d
        results[opp_name] = (w, d, l)
        vs = opp_name if opp_path is None else f"{opp_name} ({Path(opp_path).stem})"
        print(f"gen_{cycle:03d} vs {vs:20s} W/D/L {w}/{d}/{l}  score {(w + d / 2) / len(rows):.3f}   ({time.time() - start:.0f}s)",
              flush=True)
    return results


def _done(csv_path):
    if not Path(csv_path).exists():
        return set()
    with open(csv_path, newline="") as f:
        return {(int(r["cycle"]), r["opponent"]) for r in csv.DictReader(f)}


def _judged(csv_path):
    if not Path(csv_path).exists():
        return set()
    with open(csv_path, newline="") as f:
        return {int(r["cycle"]) for r in csv.DictReader(f)}


def _append(csv_path, rows, fields=FIELDS):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- standalone

def run(agent_cls, first=0, last=10**9, workers=WORKERS, **kwargs):
    """eval_checkpoint for every saved checkpoint of agent_cls with first <= cycle <= last, in order."""
    from multiprocessing import Pool

    cycles = [c for c, _ in list_checkpoints(agent_cls.CHECKPOINT_DIR) if first <= c <= last]
    with Pool(workers) as pool:
        for cycle in cycles:
            eval_checkpoint(pool, agent_cls, cycle, **kwargs)


def summary(csv_path):
    """Print score and score - expected per bucket, for each opponent and cycle."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    table = defaultdict(list)
    for r in rows:
        table[(r["opponent"], int(r["cycle"]), r["bucket"])].append(((int(r["result"]) + 1) / 2, float(r["expected"])))
    for opp in OPPONENTS:
        cycles = sorted({c for o, c, _ in table if o == opp})
        if not cycles:
            continue
        print(f"\nvs {opp}: score (score - expected) per bucket, model's view")
        print("cycle   " + "".join(f"{b:>16s}" for b in BUCKETS) + "           all")
        for c in cycles:
            cells, every = [], []
            for b in BUCKETS:
                games = table.get((opp, c, b), [])
                every += games
                if games:
                    s = sum(g[0] for g in games) / len(games)
                    e = sum(g[1] for g in games) / len(games)
                    cells.append(f"{s:6.2f} ({s - e:+.2f})")
                else:
                    cells.append("-")
            s = sum(g[0] for g in every) / len(every)
            e = sum(g[1] for g in every) / len(every)
            print(f"{c:5d}   " + "".join(f"{cell:>16s}" for cell in cells) + f"   {s:6.2f} ({s - e:+.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on the fixed eval positions.")
    parser.add_argument("--agent", required=True, help="module in Agents/AIs, e.g. leela")
    parser.add_argument("--from", dest="first", type=int, default=0, help="first cycle to evaluate")
    parser.add_argument("--to", dest="last", type=int, default=10**9, help="last cycle to evaluate")
    parser.add_argument("--greedy-every", type=int, default=GREEDY_EVERY, help="0 turns greedy off")
    parser.add_argument("--fixed-gen", type=int, default=FIXED_GEN)
    parser.add_argument("--fixed-every", type=int, default=FIXED_EVERY, help="0 turns the fixed opponent off")
    parser.add_argument("--moving-gap", type=int, default=MOVING_GAP)
    parser.add_argument("--moving-every", type=int, default=MOVING_EVERY, help="0 turns the moving opponent off")
    parser.add_argument("--greedy-positions", type=int, default=GREEDY_POSITIONS, help="positions vs greedy (each played from both sides)")
    parser.add_argument("--fixed-positions", type=int, default=FIXED_POSITIONS)
    parser.add_argument("--moving-positions", type=int, default=MOVING_POSITIONS)
    parser.add_argument("--judge-every", type=int, default=JUDGE_EVERY, help="the checkpoint scores every position itself; 0 turns it off")
    parser.add_argument("--judge-sims", type=int, default=JUDGE_SIMS, help="default: the sims the positions were scored with")
    parser.add_argument("--test-every", type=int, default=TEST_EVERY, help="print Q of the test positions' best / blunder moves; 0 turns it off")
    parser.add_argument("--test-sims", type=int, default=TEST_SIMS)
    parser.add_argument("--positions", default=None, help="eval positions file (default: Eval/eval_positions/<agent>_gen_<fixed-gen>.json)")
    parser.add_argument("--csv", default=None, help="results CSV (default: Eval/training/<agent>/<agent>_suite_<scoring checkpoint>.csv)")
    parser.add_argument("--summary", action="store_true", help="print the per-bucket table from the CSV instead of playing")
    args = parser.parse_args()

    agent_cls = importlib.import_module(f"Agents.AIs.{args.agent}").Agent
    positions = args.positions or default_positions(agent_cls, args.fixed_gen)
    if args.summary:
        summary(args.csv or default_csv(agent_cls, positions))
    else:
        run(agent_cls, args.first, args.last, positions_path=positions, csv_path=args.csv,
            greedy_every=args.greedy_every, fixed_gen=args.fixed_gen, fixed_every=args.fixed_every,
            moving_gap=args.moving_gap, moving_every=args.moving_every, greedy_positions=args.greedy_positions,
            fixed_positions=args.fixed_positions, moving_positions=args.moving_positions,
            judge_every=args.judge_every, judge_sims=args.judge_sims,
            test_every=args.test_every, test_sims=args.test_sims)
