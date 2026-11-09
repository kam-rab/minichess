"""Finds the agents in Agents/ that the UI can play against.

An agent module counts as playable if it defines either:
  * a class ``Agent`` whose instances have ``get_move(state) -> (from_sq, to_sq)``, or
  * a module-level function ``get_move(state) -> (from_sq, to_sq)``.
It can also set ``NAME = "..."`` to choose the name shown in the UI (the file name is used otherwise).
Modules that define neither, like empty stubs, are skipped. Modules that fail to import are reported.
"""

import importlib
import traceback
from pathlib import Path

HUMAN = "Human"
AGENTS_DIR = Path(__file__).resolve().parent.parent / "Agents"


def discover_agents():
    """Return ({display name: factory}, {module name: error message}).

    Calling a factory with no arguments returns an object with a ``get_move(state)`` method.
    """
    factories, errors = {}, {}
    for path in sorted(AGENTS_DIR.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"Agents.{path.stem}")
        except Exception as e:
            errors[path.stem] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            continue

        if isinstance(getattr(module, "Agent", None), type):
            factory = module.Agent
        elif callable(getattr(module, "get_move", None)):
            factory = lambda module=module: module  # the module itself acts as the agent
        else:
            continue  # not implemented yet

        name = str(getattr(module, "NAME", path.stem))
        if name == HUMAN or name in factories:
            name = f"{name} ({path.stem})"
        factories[name] = factory
    return factories, errors
