"""
Plot the eval results and training loss written by Training/loop.py.

Results live in Eval/training/<agent>/<agent>_vs_greedy.csv, <agent>_loss.csv and <agent>_games.csv,
where <agent> is the name of the agent's checkpoint folder (e.g. "l2", "unregularized").

Run from the repo root:
    python -m Eval.plot_results --agent l2                   # W/D/L vs greedy, plot once
    python -m Eval.plot_results --agent l2 --loss            # training loss instead
    python -m Eval.plot_results --agent l2 --games           # self-play game log (end reasons, conversion, endgames)
    python -m Eval.plot_results --agent l2 --material        # material vs ply scatter from the game log, by result / draw type
    python -m Eval.plot_results --agent l2 --watch           # redraw every 30 s while training runs
    python -m Eval.plot_results --agent l2 --loss --watch --interval 10
    python -m Eval.plot_results --csv path/to/file.csv --title "Some title"   # any CSV (add --loss for a loss CSV)

In --watch mode, close the plot window (or press Ctrl-C) to stop. The PNG is re-saved on every update.
The PNG is saved next to the CSV with the same name (--material saves <agent>_games_material.png).
"""

import argparse
import functools
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).resolve().parent / "training"



def _wrap_to_width(text, text_obj, width, renderer):
    """Re-wrap text (keeping its existing line breaks) so no line is wider than width pixels."""
    prop = text_obj.get_fontproperties()
    measure = lambda line: renderer.get_text_width_height_descent(line, prop, ismath=False)[0]
    out = []
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            candidate = f"{line} {word}" if line else word
            if line and measure(candidate) > width:
                out.append(line)
                line = word
            else:
                line = candidate
        out.append(line)
    return "\n".join(out)


def _tight_layout_wrapped(fig):
    """tight_layout, after wrapping each subplot title to its subplot's width and the figure title to the figure's."""
    fig.tight_layout()
    renderer = fig.canvas.get_renderer() if hasattr(fig.canvas, "get_renderer") else fig._get_renderer()
    changed = False
    titles = [(ax.title, ax.get_window_extent(renderer).width) for ax in fig.axes if ax.title.get_text()]
    if fig._suptitle is not None:
        titles.append((fig._suptitle, fig.bbox.width * 0.95))
    for t, width in titles:
        original = t.get_text()
        wrapped = _wrap_to_width(original, t, width, renderer)
        if wrapped != original:
            t.set_text(wrapped)
            changed = True
    if changed:
        fig.tight_layout()     # wrapped titles are taller: lay out again


def agent_paths(agent, loss=False, games=False):
    """(csv path, default title) for an agent's results, matching the paths Training/loop.py writes."""
    if games:
        return RESULTS_DIR / agent / f"{agent}_games.csv", f"{agent} self-play games"
    if loss:
        return RESULTS_DIR / agent / f"{agent}_loss.csv", f"{agent} training loss (mean over each cycle)"
    return RESULTS_DIR / agent / f"{agent}_vs_greedy.csv", f"{agent} vs 4-ply greedy"


def load_results(csv_path):
    """Read a results CSV. Returns (cycle, win_rate, draw_rate, loss_rate) as arrays sorted by cycle."""
    with open(csv_path, newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["cycle"]))
    cycle = np.array([int(r["cycle"]) for r in rows])
    games = np.array([int(r["games"]) for r in rows])
    w, d, l = (np.array([int(r[k]) for r in rows]) / games for k in ("wins", "draws", "losses"))
    return cycle, w, d, l


def draw(fig, csv_path, title):
    """Clear fig and draw stacked win/draw/loss rates (top) and score = W + 0.5*D (bottom)."""
    cycle, w, d, l = load_results(csv_path)

    fig.clear()
    ax1, ax2 = fig.subplots(2, 1, sharex=True)

    # top: win / draw / loss rates stacked so they sum to 1
    ax1.stackplot(cycle, w, d, l, labels=["Win", "Draw", "Loss"],
                  colors=["#4caf50", "#9e9e9e", "#e57373"], alpha=0.85)
    ax1.set_ylabel("Fraction of games")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left")
    ax1.set_title(f"{title}\n(latest: cycle {cycle[-1]})" if len(cycle) else title)

    # bottom: score = (W + 0.5 D) / N, with 0.5 = even with the opponent
    ax2.plot(cycle, w + 0.5 * d, marker="o")
    ax2.axhline(0.5, linestyle="--", color="gray")
    ax2.set_ylabel("Score (W + ½D)")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Training cycle (200 steps each)")

    _tight_layout_wrapped(fig)


def _float_or_nan(x):
    """CSV cell -> float, or nan if the cell is missing/empty (e.g. rows logged before a column existed)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load_loss(csv_path):
    """
    Read a loss CSV. Returns (cycle, policy_ce, value_mse, ce_floor, moves_left) as arrays sorted by cycle.
    ce_floor and moves_left (the moves-left head's MSE, agents with that head only) are nan for rows that don't have them.
    """
    with open(csv_path, newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["cycle"]))
    cycle = np.array([int(r["cycle"]) for r in rows])
    policy = np.array([float(r["policy_ce"]) for r in rows])
    value = np.array([float(r["value_mse"]) for r in rows])
    floor = np.array([_float_or_nan(r.get("ce_floor")) for r in rows])
    moves_left = np.array([_float_or_nan(r.get("moves_left_loss")) for r in rows])
    return cycle, policy, value, floor, moves_left


def draw_loss(fig, csv_path, title):
    """
    Clear fig and draw policy CE (left axis) and value MSE (right axis) against training cycle.
    If the CSV has ce_floor (entropy of the policy targets), it is drawn as a dashed line under policy CE:
    the gap between the two is how far the network is from matching its targets.
    If the CSV has moves_left_loss (agents with a moves-left head), it is drawn on the right axis with value MSE.
    """
    cycle, policy, value, floor, moves_left = load_loss(csv_path)

    fig.clear()
    ax1 = fig.subplots()
    ax2 = ax1.twinx()                      # second y-axis: the two losses are on different scales

    l1, = ax1.plot(cycle, policy, color="tab:blue", label="Policy CE")
    l2, = ax2.plot(cycle, value, color="tab:orange", label="Value MSE")
    handles = [l1, l2]
    has_floor = np.isfinite(floor)
    if has_floor.any():
        l3, = ax1.plot(cycle[has_floor], floor[has_floor], color="tab:blue", linestyle="--",
                       label="Policy CE floor (target entropy)")
        handles.insert(1, l3)
    has_moves_left = np.isfinite(moves_left)
    if has_moves_left.any():
        l4, = ax2.plot(cycle[has_moves_left], moves_left[has_moves_left], color="tab:green",
                       label="Moves-left MSE (plies left / 160)")
        handles.append(l4)
    ax1.set_ylabel("Policy cross-entropy", color="tab:blue")
    ax2.set_ylabel("Value MSE" + (" / moves-left MSE" if has_moves_left.any() else ""), color="tab:orange")
    ax1.tick_params(axis="y", colors="tab:blue")
    ax2.tick_params(axis="y", colors="tab:orange")
    ax1.set_xlabel("Training cycle (200 steps each)")
    ax1.legend(handles=handles, loc="upper right")
    ax1.set_title(f"{title}\n(latest: cycle {cycle[-1]})" if len(cycle) else title)

    _tight_layout_wrapped(fig)


GAMES_WINDOW = 10      # cycles per rolling window in the game-log plot (50 games per cycle is noisy)
def _winner_material(r):
    """Final material from the winner's view; for draws, the size of the lead of whichever side was ahead."""
    mat = int(r["final_material"])
    return mat * r["result"] if r["result"] != 0 else abs(mat)


def _is(reason, lo=None, hi=None):
    """Test for one band of the end-of-game plot: end reason, and winner's material in [lo, hi]."""
    def test(r):
        if r.get("end_reason") != reason:
            return False
        m = _winner_material(r)
        return (lo is None or m >= lo) and (hi is None or m <= hi)
    return test


# bands of the "how games end" panel, bottom to top: (label, test, color)
END_BANDS = [
    ("mate, up 4+",          _is("checkmate", 4),        "#1b5e20"),
    ("mate, up 2-3",         _is("checkmate", 2, 3),     "#43a047"),
    ("mate, even (-1 to 1)", _is("checkmate", -1, 1),    "#81c784"),
    ("mate, down 2+",        _is("checkmate", None, -2), "#c8e6c9"),
    ("threefold, up 4+",     _is("threefold", 4),        "#b71c1c"),
    ("threefold, up 2-3",    _is("threefold", 2, 3),     "#e53935"),
    ("threefold, even",      _is("threefold", 0, 1),     "#ef9a9a"),
    ("stalemate",            _is("stalemate"),           "#ffb74d"),
    ("insufficient",         _is("insufficient"),        "#9e9e9e"),
    ("80-move",              _is("80-move"),             "#7986cb"),
]
END_BAND_LABELS = [b[0] for b in END_BANDS]
END_BAND_TESTS = [b[1] for b in END_BANDS]
END_BAND_COLORS = [b[2] for b in END_BANDS]


LEAD_THRESHOLDS = [2, 4, 6, 8]             # "once a side is up N or more" lines in the plies-to-win panel
LEAD_COLORS = ["#a5d6a7", "#66bb6a", "#2e7d32", "#0b3d12"]


def _plies_to_win(r):
    """
    {N: plies from the first ply where one side led by >= N (material unchanged for 2 plies, so not mid-trade)
    to the end of the game, if that side went on to win}. Thresholds never reached, or not won, are left out.
    """
    traj = [int(x) for x in r["material"].split()]
    out = {}
    for n in LEAD_THRESHOLDS:
        for i in range(2, len(traj)):
            if abs(traj[i]) >= n and traj[i] == traj[i - 1] == traj[i - 2]:
                if r["result"] == np.sign(traj[i]):
                    out[n] = r["plies"] - i
                break
    return out


def load_games(csv_path):
    """Read a game-log CSV (one row per self-play game). Returns a list of dicts with numeric fields converted."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("cycle", "result", "plies", "first_big_lead_ply", "big_lead_side", "endgame_plies", "final_pieces"):
            r[k] = int(r[k])
        r["plies_to_win"] = _plies_to_win(r)
    return rows


def _trailing_sum(x, window):
    """Sum of x over the last `window` entries at each index (shorter at the start)."""
    return np.convolve(x, np.ones(window))[:len(x)]


def draw_games(fig, csv_path, title, window=GAMES_WINDOW):
    """
    Clear fig and draw the self-play game log, each point pooled over the last `window` cycles:
      top left:     how games end (share of each end reason)
      top right:    big-lead games: share of games with a big lead, and what share of those were won / drawn / lost
      bottom left:  endgame share of positions (left axis) and mean game length (right axis)
      bottom right: median plies from first being up N+ (N in LEAD_THRESHOLDS) to the win, for games that side won
    """
    rows = load_games(csv_path)
    cycles = np.array(sorted({r["cycle"] for r in rows}))
    by_cycle = {c: [] for c in cycles}
    for r in rows:
        by_cycle[r["cycle"]].append(r)

    def per_cycle(fn):
        return np.array([fn(by_cycle[c]) for c in cycles], dtype=float)

    def pooled(numerator, denominator):    # ratio of window sums, not mean of per-cycle ratios
        num, den = _trailing_sum(numerator, window), _trailing_sum(denominator, window)
        return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

    n_games = per_cycle(len)

    fig.clear()
    (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2, sharex=True)

    # how games end, split by the final material of the side that won (draws: the side that was ahead)
    shares = [pooled(per_cycle(lambda g, f=f: sum(f(r) for r in g)), n_games) for f in END_BAND_TESTS]
    ax1.stackplot(cycles, *shares, labels=END_BAND_LABELS, colors=END_BAND_COLORS, alpha=0.9)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Fraction of games")
    ax1.set_title("How games end (by final material of the winner / side ahead)")
    ax1.legend(loc="upper right", bbox_to_anchor=(-0.18, 1), fontsize=7)   # outside, left of the panel

    # big-lead conversion
    leads = per_cycle(lambda g: sum(r["big_lead_side"] != 0 for r in g))
    won = per_cycle(lambda g: sum(r["big_lead_side"] != 0 and r["result"] == r["big_lead_side"] for r in g))
    drawn = per_cycle(lambda g: sum(r["big_lead_side"] != 0 and r["result"] == 0 for r in g))
    lost = per_cycle(lambda g: sum(r["big_lead_side"] != 0 and r["result"] == -r["big_lead_side"] for r in g))
    ax2.plot(cycles, pooled(won, leads), color="#4caf50", label="lead converted (won)")
    ax2.plot(cycles, pooled(drawn, leads), color="#9e9e9e", label="lead drawn")
    ax2.plot(cycles, pooled(lost, leads), color="#e57373", label="lead lost")
    ax2.plot(cycles, pooled(leads, n_games), color="black", linestyle="--", label="games with a big lead")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Fraction")
    ax2.set_title("Games where one side got a big lead")
    ax2.legend(loc="lower left", fontsize=8)

    # endgame share and game length
    endgame = pooled(per_cycle(lambda g: sum(r["endgame_plies"] for r in g)), per_cycle(lambda g: sum(r["plies"] for r in g)))
    length = pooled(per_cycle(lambda g: sum(r["plies"] for r in g)), n_games)
    l1, = ax3.plot(cycles, endgame, color="tab:purple", label="endgame share of positions")
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("Endgame share", color="tab:purple")
    ax3b = ax3.twinx()
    l2, = ax3b.plot(cycles, length, color="tab:brown", label="mean game length (plies)")
    # 25th / 75th percentile of game length, pooled over the window like everything else
    window_plies = [[r["plies"] for c in cycles[max(0, i - window + 1):i + 1] for r in by_cycle[c]] for i in range(len(cycles))]
    p25 = [np.percentile(p, 25) for p in window_plies]
    p75_len = [np.percentile(p, 75) for p in window_plies]
    l3, = ax3b.plot(cycles, p25, color="tab:brown", linestyle=":", label="game length 25th / 75th percentile")
    ax3b.plot(cycles, p75_len, color="tab:brown", linestyle=":")
    ax3b.set_ylim(bottom=0)
    ax3b.set_ylabel("Plies", color="tab:brown")
    ax3.set_title("Endgames and game length")
    ax3.legend(handles=[l1, l2, l3], loc="upper left", fontsize=8)
    ax3.set_xlabel("Training cycle")

    # median plies from first being up >= N to the win, one line per N, pooled over the window
    for n, color in zip(LEAD_THRESHOLDS, LEAD_COLORS):
        med = []
        for i in range(len(cycles)):
            gaps = [r["plies_to_win"][n] for c in cycles[max(0, i - window + 1):i + 1] for r in by_cycle[c] if n in r["plies_to_win"]]
            med.append(np.median(gaps) if gaps else np.nan)
        ax4.plot(cycles, med, color=color, label=f"up {n}+")
    ax4.set_ylim(bottom=0)
    ax4.set_ylabel("Median plies")
    ax4.set_title("Plies from first being up N+ to the win (shuffling)")
    ax4.legend(loc="lower left", fontsize=8, ncol=4)
    ax4.set_xlabel("Training cycle")

    latest = f"\n(latest: cycle {cycles[-1]}, each point = last {window} cycles)" if len(cycles) else ""
    fig.suptitle(title + latest)
    _tight_layout_wrapped(fig)


MATERIAL_SAMPLE = 3000     # normal-start games sampled for the win panels of the material scatter
MATERIAL_CAP = 1500        # most games drawn in any one panel, so dense panels stay readable


def draw_material(fig, csv_path, title, since=None, last=None):
    """
    Clear fig and draw material (White's view) against ply for normal-start games, one dot per position,
    jittered so stacked points show up. Six panels: White won, Black won, and one per draw type.
    The win panels use a random sample of MATERIAL_SAMPLE games; the draw panels use every draw,
    so the rare draw types have enough points. Panels with more than MATERIAL_CAP games show a random subset.
    Only games from cycle `since` on are used, or from the newest `last` cycles (both None: the whole log).
    """
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("start", "normal") == "normal"]
    if last is not None and rows:
        since = max(int(r["cycle"]) for r in rows) - last + 1      # recomputed each redraw, so it follows --watch
    if since is not None:
        rows = [r for r in rows if int(r["cycle"]) >= since]
    rng = np.random.default_rng(0)                  # fixed seed so redraws in --watch mode don't flicker
    sample = [rows[i] for i in rng.choice(len(rows), min(MATERIAL_SAMPLE, len(rows)), replace=False)]
    draws = [r for r in rows if r["result"] == "0"]
    panels = [("White won", lambda r: r["result"] == "1", sample, "#2e7d32"),
              ("Black won", lambda r: r["result"] == "-1", sample, "#c62828"),
              ("Draw: threefold", lambda r: r.get("end_reason") == "threefold", draws, "#b71c1c"),
              ("Draw: insufficient material", lambda r: r.get("end_reason") == "insufficient", draws, "#616161"),
              ("Draw: stalemate", lambda r: r.get("end_reason") == "stalemate", draws, "#ef6c00"),
              ("Draw: 80-move rule", lambda r: r.get("end_reason") == "80-move", draws, "#3949ab")]

    fig.clear()
    axes = fig.subplots(2, 3, sharex=True, sharey=True)
    for ax, (label, test, pool, color) in zip(axes.flat, panels):
        games = [r for r in pool if test(r)]
        total = len(games)
        if total > MATERIAL_CAP:
            games = [games[i] for i in rng.choice(total, MATERIAL_CAP, replace=False)]
        if games:
            ys = [np.array(r["material"].split(), dtype=float) for r in games]
            xs = [np.arange(len(y)) for y in ys]
            n = sum(map(len, xs))
            ax.scatter(np.concatenate(xs) + rng.uniform(-0.4, 0.4, n), np.concatenate(ys) + rng.uniform(-0.35, 0.35, n),
                       s=1.5, alpha=0.08 if len(games) > 500 else 0.25, color=color, rasterized=True)
        ax.axhline(0, color="black", lw=0.5)
        if pool is sample:     # win panels: from the sample, which is every game when there are few enough
            nb = "\u00a0"         # non-breaking spaces keep the parenthetical on one line when the title wraps
            source = (f"of {len(sample):,} games (sampled{nb}from{nb}{len(rows):,})" if len(sample) < len(rows)
                      else f"of {len(rows):,} games")
        else:
            source = f"of {len(draws):,} draws"
        shown = f", {len(games):,}\u00a0shown" if len(games) < total else ""
        ax.set_title(f"{label}\n{total:,} {source}{shown}", fontsize=10)
        ax.set_xlim(0, 162)
        ax.set_ylim(-30, 30)
    for ax in axes[1]:
        ax.set_xlabel("Ply (move number)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Material (White's view)")
    cycles = [int(r["cycle"]) for r in rows]
    window = f"newest {last} cycles: " if last is not None else f"cycle {since} on: " if since is not None else ""
    span = (f"\n(normal-start games, {window}cycles {min(cycles)}-{max(cycles)}, {len(rows):,} games, jittered)"
            if cycles else f"\n(no normal-start games in {window.rstrip(': ') or 'the log'})")
    fig.suptitle(title + span)
    _tight_layout_wrapped(fig)


FIGSIZES = {draw_games: (14, 8), draw_material: (20, 11)}     # anything else: (9, 7)


def _figsize(draw_fn):
    return FIGSIZES.get(getattr(draw_fn, "func", draw_fn), (9, 7))     # getattr: draw_fn may be a functools.partial


def plot_results(csv_path, title, out_path=None, show=True, draw_fn=draw):
    """Plot once; optionally save to out_path and/or show a window."""
    fig = plt.figure(figsize=_figsize(draw_fn))
    draw_fn(fig, csv_path, title)
    if out_path:
        fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def watch(csv_path, title, out_path=None, interval=30, draw_fn=draw):
    """
    Redraw whenever the CSV changes, checking every `interval` seconds, until the window is closed.
    Runs in its own process, so training is never slowed down or blocked by the plot.
    """
    csv_path = Path(csv_path)
    fig = plt.figure(figsize=_figsize(draw_fn))
    plt.show(block=False)
    last_mtime = None

    while plt.fignum_exists(fig.number):          # stops when the window is closed
        mtime = csv_path.stat().st_mtime if csv_path.exists() else None
        if mtime is not None and mtime != last_mtime:
            try:
                draw_fn(fig, csv_path, title)
                fig.canvas.draw_idle()
                if out_path:
                    fig.savefig(out_path, dpi=150)
                last_mtime = mtime
            except (ValueError, KeyError, IndexError):
                pass                              # header-only or half-written file: try again next time
        plt.pause(interval)                       # keeps the window responsive while waiting


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot eval results (W/D/L vs training cycle).")
    parser.add_argument("--agent", default=None,
                        help="agent folder name, e.g. l2 or Unregularized (picks the CSV and title)")
    parser.add_argument("--loss", action="store_true", help="plot training loss instead of W/D/L")
    parser.add_argument("--games", action="store_true", help="plot the self-play game log instead of W/D/L")
    parser.add_argument("--material", action="store_true", help="plot material vs ply from the game log")
    parser.add_argument("--since", type=int, default=None, help="--material: only games from this cycle on")
    parser.add_argument("--last", type=int, default=None, help="--material: only games from the newest N cycles")
    parser.add_argument("--csv", default=None, help="CSV to plot (overrides --agent)")
    parser.add_argument("--title", default=None, help="plot title (overrides the --agent default)")
    parser.add_argument("--watch", action="store_true", help="redraw as new rows are added")
    parser.add_argument("--interval", type=float, default=30, help="seconds between checks in --watch mode")
    args = parser.parse_args()

    if args.agent is None and args.csv is None:
        available = sorted(p.name for p in RESULTS_DIR.iterdir() if p.is_dir()) if RESULTS_DIR.exists() else []
        parser.error(f"pass --agent (available: {', '.join(available) or 'none yet'}) or --csv")

    if (args.since is not None or args.last is not None) and not args.material:
        parser.error("--since / --last only apply to --material")
    if args.since is not None and args.last is not None:
        parser.error("pass --since or --last, not both")
    draw_fn = draw_material if args.material else draw_games if args.games else draw_loss if args.loss else draw
    if args.material:
        draw_fn = functools.partial(draw_material, since=args.since, last=args.last)
    agent_csv, agent_title = agent_paths(args.agent, args.loss, args.games or args.material) if args.agent else (None, None)
    if args.material and agent_title:
        agent_title = f"{args.agent} material vs ply"
    csv_path = Path(args.csv) if args.csv else agent_csv
    title = args.title or agent_title or csv_path.stem
    # same name as the CSV; the material plot gets its own suffix so it doesn't overwrite <agent>_games.png
    # a filtered plot gets its own name too, so it doesn't overwrite the full one
    window = f"_since{args.since}" if args.since is not None else f"_last{args.last}" if args.last is not None else ""
    out_path = csv_path.with_name(csv_path.stem + f"_material{window}.png") if args.material else csv_path.with_suffix(".png")
    try:
        if args.watch:
            watch(csv_path, title, out_path, args.interval, draw_fn)
        else:
            plot_results(csv_path, title, out_path, draw_fn=draw_fn)
    except KeyboardInterrupt:
        pass
