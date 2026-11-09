# minichess
A homemade alpha-zero style neural network to play 5x5 minichess

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repo Structure

```
minichess
|
|-Agents/
|  |-mcts.py  
|  |-rand.py
|  |-greedy_material.py
|
|-Engine/
|  |-state.py
|  |-pieces.py
|
|-Eval/
|  |-tourney.py
|  |-plot_results.py
|
|-Model/
|  |-layers.py
|  |-loss.py
|  |-model.py
|  |-regularization.py
|
|-Training/
|  |-loop.py
|
|-UI/
|  |-__main__.py     # entry point: python -m UI
|  |-app.py          # window, game control, running agents
|  |-board_view.py   # drawing the board and pieces
|  |-widgets.py      # buttons and player selectors
|  |-players.py      # finds playable agents in Agents/
|
|-requirements.txt
```

## Playing

Run from the repo root:

```bash
python -m UI                                  # human vs human
python -m UI --black rand                     # human (White) vs an agent
python -m UI --white rand --black mcts        # watch two agents play
python -m UI --state "<State.state_to_string() output>"   # start from a custom position
```

Click a piece to see its legal moves, then click a destination square. Each side's player can be changed at any
time in the side panel. Keys: `N` new game, `U`/Backspace undo, `F` flip board, Space pause/resume agents,
Esc deselect.

An agent shows up in the UI once its module in `Agents/` defines either a class `Agent` with a
`get_move(self, state)` method, or a module-level `get_move(state)` function. Either one gets a copy of the
current `State` and returns `(from_sq, to_sq)`. Set `NAME = "..."` in the module to change its display name.
