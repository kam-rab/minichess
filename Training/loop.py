import numpy as np
from pathlib import Path
import csv
from multiprocessing import Pool
from Training.ring_buffer import RingBuffer
from Training.self_play import self_play_games
from Eval.eval_suite import eval_checkpoint
from Eval.tourney import list_checkpoints, append_row
from Agents.greedy_material import Agent as Greedy


# training hyperparams
NUM_GAMES_BEFORE_TRAINING = 100
TRAINING_STEPS_PER_CYCLE = 200
NUM_CYCLES = 1001

LR_SCHEDULE = [(0, 1e-3), (2475, 3e-4), (4000, 1e-4)]

# endgame starts: some games start from endgame positions reached in earlier self-play games
ENDGAME_FRACTION = 0.25         # share of each cycle's games started from a saved endgame position
ENDGAME_POOL_SIZE = 2000        # oldest positions drop out, so the pool follows the current agent
ENDGAME_POOL_MIN = 50           # only start endgame games once the pool has this many positions

# eval params; don't affect training
NUM_EVAL_GAMES = 50
NUM_CYCLES_BEFORE_EVAL = 25
EVAL_RESULTS_DIR = Path(__file__).resolve().parent.parent / "Eval" / "training"

NUM_WORKERS = 6     # run more cores on my machine to generate self_play data


def train(agent_class, resume_training=False, num_cycles=NUM_CYCLES, workers=NUM_WORKERS, lr_schedule = LR_SCHEDULE):
    """
    Training loop for each AI agent. Run from main.py, passing in agent object. 
    """
    trainee = agent_class(load_checkpoint=resume_training)
    adam_path = agent_class.ADAM_PATH       # load saved adam if they exist for faster/better training
    if resume_training and adam_path is not None and adam_path.exists():
        trainee.model.load_adam(str(adam_path))
    has_moves_left = len(trainee.model.leaves) > 2      # leela
    x_dim = trainee.model.get_layer_type()[0].weights.shape[0] - 1    # get x dim from model - bias row

    agent_class.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)       # make save dir if doesn't already exist
    LATEST = agent_class.CHECKPOINT_DIR / "latest.npz"
    name = agent_class.CHECKPOINT_DIR.name                     # "l2" / "unregularized"
    LOSS_CSV = EVAL_RESULTS_DIR / name / f"{name}_loss.csv"
    GAMES_CSV = EVAL_RESULTS_DIR / name / f"{name}_games.csv"
    EVAL_CSV = EVAL_RESULTS_DIR / name / f"{name}_vs_greedy.csv"

    ckpts = list_checkpoints(agent_class.CHECKPOINT_DIR)
    if not resume_training and ckpts:
        raise RuntimeError(f"{agent_class.CHECKPOINT_DIR} already has checkpoints; pass resume_training=True or delete them")
    if not resume_training:
        for path in (LOSS_CSV, EVAL_CSV, GAMES_CSV):
            path.unlink(missing_ok=True)
    start = ckpts[-1][0] + 1 if (resume_training and ckpts) else 0      # for save logic, ensure saved checkpoints don't overwrite old ones

    rng = np.random.default_rng()
    ring_buffer = RingBuffer()
    BUFFER_PATH = agent_class.CHECKPOINT_DIR / "buffer.npz"
    ENDGAME_POOL_PATH = agent_class.CHECKPOINT_DIR / "endgame_pool.npy"
    endgame_pool = []                             # State strings of endgame positions, oldest first
    if resume_training:
        if BUFFER_PATH.exists():                  # resume from previous buffer
            ring_buffer.load(str(BUFFER_PATH))
        if ENDGAME_POOL_PATH.exists():
            endgame_pool = list(np.load(ENDGAME_POOL_PATH))


    # start cycle with workers generating self play games and running evals
    with Pool(workers) as pool:
        for cycle in range(start, num_cycles):
            # update the learning rate according to the schedule
            lr = [lr for start, lr in lr_schedule if cycle >= start][-1]
            trainee.model.update_lr(lr)
            print("CYCLE: ", cycle, " // LEARNING RATE: ", lr)

            trainee.model.save(str(LATEST))          # publish current weights to the workers
            cycle_split = [0, 0, 0]
            cycle_stats = []

            # start some games from saved endgame positions, the rest from the opening
            n_end = int(ENDGAME_FRACTION * NUM_GAMES_BEFORE_TRAINING) if len(endgame_pool) >= ENDGAME_POOL_MIN else 0
            picks = rng.choice(len(endgame_pool), n_end, replace=False) if n_end else []
            starts = [None] * (NUM_GAMES_BEFORE_TRAINING - n_end) + [str(endgame_pool[i]) for i in picks]

            for X, pi, v, moves_left, mask, q, term, stats in self_play_games(pool, agent_class, str(LATEST), cycle, NUM_GAMES_BEFORE_TRAINING, cycle * 100_000, starts):
                ring_buffer.add(X, pi, v, mask, moves_left=moves_left, q=q)
                cycle_split[term + 1] += 1
                cycle_stats.append(stats)
                if stats["endgame_start"] is not None:          # only normal games save one
                    endgame_pool.append(stats["endgame_start"]["state"])
            endgame_pool = endgame_pool[-ENDGAME_POOL_SIZE:]

            training_data = ring_buffer.get_data(has_moves_left, q=True)     # value target (z + q) / 2
            losses = [trainee.train(*training_data, return_loss=True) for _ in range(TRAINING_STEPS_PER_CYCLE)]
            losses = np.mean(losses, axis=0)

            # also draw floor for policy loss:
            X, (pi, v, moves_left), _ = training_data
            ce_floor = -np.sum(pi * np.log(np.where(pi > 0, pi, 1))) / len(pi)

            # print stats during training
            loss_printout = ""
            loss_names = ["policy CE", "value MSE", "moves_left_MSE"]
            for i in range(len(losses)):
                loss_printout += loss_names[i] + f": {losses[i]:.4f}   "    

            print(loss_printout)
            print("White win rate: ", cycle_split[2]/sum(cycle_split))
            print("Draw rate: ", cycle_split[1]/sum(cycle_split))
            print("Black win rate: ", cycle_split[0]/sum(cycle_split))
            print(f"Endgame starts: {n_end}   endgame pool: {len(endgame_pool)}")
            print("\n")


            # save checkpoint
            trainee.model.save(str(agent_class.CHECKPOINT_DIR / f"gen_{cycle:03d}.npz"))
            if agent_class.ADAM_PATH is not None:
                trainee.model.save_adam(str(agent_class.ADAM_PATH))
            ring_buffer.save(str(BUFFER_PATH))  
            np.save(ENDGAME_POOL_PATH, np.array(endgame_pool, dtype=str))
            append_loss_row(LOSS_CSV, cycle, losses[0], losses[1], ce_floor, *losses[2:])
            append_game_rows(GAMES_CSV, cycle, cycle_stats)

            # delete previous checkpoint for storage space
            prev = agent_class.CHECKPOINT_DIR / f"gen_{cycle - 1:03d}.npz"
            if (cycle - 1) % NUM_CYCLES_BEFORE_EVAL != 0:
                prev.unlink(missing_ok=True)
    
            # eval againts fixed opponents occasinoally
            # fixed is gen_1000 leela, moving is 100 cycles previous from current gen. 
            # 25 games (each side) against greedy, 50 from each side against both fixed and moving
            eval_checkpoint(pool, agent_class, cycle, greedy_every=50, greedy_positions=25, fixed_gen=1000, fixed_every=100, fixed_positions=50,
                            moving_gap=100, moving_every=100, moving_positions=50)



def append_loss_row(path, cycle, policy_loss, value_loss, ce_floor, moves_left_loss=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["cycle", "policy_ce", "value_mse", "ce_floor", "moves_left_loss"])
        if moves_left_loss is not None:
            writer.writerow([cycle, policy_loss, value_loss, ce_floor, moves_left_loss])
        else:
            writer.writerow([cycle, policy_loss, value_loss, ce_floor])


GAME_FIELDS = ["start", "result", "end_reason", "plies", "max_white_lead", "max_black_lead", "first_big_lead_ply", "big_lead_side", "endgame_plies", "final_material", "final_pieces"]
def append_game_rows(path, cycle, cycle_stats):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["cycle"] + GAME_FIELDS + ["material"])
        for s in cycle_stats:
            writer.writerow([cycle] + [s[k] for k in GAME_FIELDS] + [" ".join(map(str, s["material"]))])



# Update code for each run. Used for testing architecture and loss convergence for certain models, given static ring buffer data. 
def comparison():
    """
    Comparison between models for loss convergence on static ring buffer self play data.
    """
    extra_layer = Unregularized(load_checkpoint=False, extra_layer=True)  # used for static comparison on ring buffer data between architectures
    current = Unregularized(load_checkpoint=False)                        # if want to test a new architecture, update signature again of Agent.__init__()

    ring_buffer = RingBuffer()
    BUFFER_PATH = CHECKPOINT_DIR / "buffer.npz"
    if BUFFER_PATH.exists():                  # resume from previous buffer
        ring_buffer.load(str(BUFFER_PATH))

    rng = np.random.default_rng()
    X, y, masks = ring_buffer.get_data()
    pi, v = y
    indices = rng.permutation(len(X))
    hold, keep = indices[:len(indices)//10], indices[len(indices)//10:]
    holdout_X, holdout_y, holdout_masks = X[hold], (pi[hold], v[hold]), masks[hold]
    X, y, masks = X[keep], (pi[keep], v[keep]), masks[keep]

    for step in range(5001):
        if step % 500 == 0: 
            print("\n")
            print("Extra layer holdout loss: ", extra_layer.evaluate(holdout_X, holdout_y, holdout_masks))
            print("Current architecture loss: ", current.evaluate(holdout_X, holdout_y, holdout_masks))
            print("\n")
            print(extra_layer.train(X, y, masks, return_loss=True))
            print(current.train(X, y, masks, return_loss=True))

        extra_layer.train(X, y, masks, print_loss)
        current.train(X, y, masks, print_loss)

    print("CE floor:", -np.sum(holdout_y[0] * np.log(np.where(holdout_y[0] > 0, holdout_y[0], 1))) / len(holdout_y[0]), " MSE baseline:", np.mean(holdout_y[1]**2))



