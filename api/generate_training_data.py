import numpy as np

from api.define_game import make_move, check_win, check_draw


def generate_training_data1(num_games):
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

            game_states.append(board.copy())
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

    return np.array(X), np.array(y)

# modify the generate_training_data function to convert the game states to numerical values
# modify the generate_training_data function to convert the optimal moves to numerical values
def generate_training_data(num_games):
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

            game_states.append(board.copy())
            optimal_moves.append(move)

            if check_win(board, current_player):
                outcome = 1 if current_player == 'X' else -1
                break
            elif check_draw(board):
                outcome = 0
                break

            current_player = 'O' if current_player == 'X' else 'X'

        for state, move in zip(game_states, optimal_moves):
            X.append([1 if cell == 'X' else -1 if cell == 'O' else 0 for cell in state])
            y.append(move)

    return np.array(X), np.array(y)
