import numpy as np

from api.define_game import make_move, check_win, check_draw
from api.get_human_move import get_human_move
from api.predict_move import predict_move
from api.print_board import print_board
from api.train_neural_network import model


def play_game_against_ai(model):
    """
    Play a game against the AI.
    :param model: Trained neural network model.
    """
    board = ['-' for _ in range(9)]
    current_player = 'O'

    while True:
        print_board(board)

        if current_player == 'X':
            # Human player's turn
            move = get_human_move(board)
        else:
            # AI player's turn
            predictions = predict_move(board, model)
            sorted_moves = np.argsort(predictions)[::-1]  # Sort moves in descending order of prediction
            for move in sorted_moves:
                if board[move] == '-':
                    break
            print(f"Predicted optimal move: {move + 1}")  # +1 to convert to 1-based index
            print(f"AI plays at position {move + 1}")

        if board[move] == '-':
            make_move(board, move, current_player)

        if check_win(board, current_player):
            print_board(board)
            print(f"{current_player} wins!")
            break
        elif check_draw(board):
            print_board(board)
            print("It's a draw!")
            break

        # Switch to the other player
        current_player = 'O' if current_player == 'X' else 'X'


if __name__ == '__main__':
    play_game_against_ai(model)
