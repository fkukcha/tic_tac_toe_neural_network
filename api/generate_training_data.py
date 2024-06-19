import numpy as np

from api.convert_board_to_numeric import convert_board_to_numeric
from api.define_game import make_move, check_win, check_draw


def generate_training_data(num_games):
    """
    Generates training data for the neural network.
    :param num_games: Number of games to play.
    :return: X, y: Training data and labels.
    """
    X = []
    y = []
    for _ in range(num_games):
        board = ['-' for _ in range(9)]
        game_states = []
        optimal_moves = []

        current_player = 'X'
        while True:
            available_moves = [i for i, cell in enumerate(board) if cell == '-']
            move = np.random.choice(available_moves)
            make_move(board, move, current_player)

            game_states.append(convert_board_to_numeric(board))
            optimal_moves.append(move)

            if check_win(board, current_player):
                outcome = 1 if current_player == 'X' else -1
                break
            elif check_draw(board):
                outcome = 0
                break

            current_player = 'O' if current_player == 'X' else 'X'

        for state, move in zip(game_states, optimal_moves):
            X.append(state)
            label = np.zeros(9)
            label[move] = outcome
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
