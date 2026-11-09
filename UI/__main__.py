"""Launch the minichess UI from the repo root:  python -m UI [--white NAME] [--black NAME] [--state STATE]"""

import argparse

from Engine.state import STATE_INIT
from UI.app import App
from UI.players import HUMAN


def main():
    parser = argparse.ArgumentParser(description="Play minichess against humans or agents from Agents/.")
    parser.add_argument("--white", default=HUMAN, help="player for White: 'Human' or an agent name (default: Human)")
    parser.add_argument("--black", default=HUMAN, help="player for Black: 'Human' or an agent name (default: Human)")
    parser.add_argument("--state", default=STATE_INIT,
                        help="starting position as an engine state string, as produced by State.state_to_string()")
    args = parser.parse_args()
    App(start_state=args.state, white=args.white, black=args.black).run()


if __name__ == "__main__":
    main()
