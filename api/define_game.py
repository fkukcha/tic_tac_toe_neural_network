def check_win(board, player):
    """
    Checks if the current player has won.
    :param board: Current board state.
    :param player: Current player ('X' or 'O').
    :return: True if the player has won, False otherwise.
    """
    win_conditions = [
        [board[0], board[1], board[2]],
        [board[3], board[4], board[5]],
        [board[6], board[7], board[8]],
        [board[0], board[3], board[6]],
        [board[1], board[4], board[7]],
        [board[2], board[5], board[8]],
        [board[0], board[4], board[8]],
        [board[2], board[4], board[6]]
    ]
    return [player, player, player] in win_conditions


def check_draw(board):
    """
    Checks if the board is full (draw).
    :param board: Current board state.
    :return: True if the board is full, False otherwise.
    """
    return all(cell != '-' for cell in board)


def make_move(board, position, player):
    """
    Makes a move on the board.
    :param board: Current board state.
    :param position: Position to make the move.
    :param player: Current player ('X' or 'O').
    """
    board[position] = player
