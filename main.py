from Engine import pieces, state


game = state.State()
print(game.state_to_string())
game.play_move(((3, 1), (2, 1)))
print(game.state_to_string())
game.play_move((2, 1))


