from api.define_game import check_win, check_draw, make_move


def test_check_win():
    """
    Test the check_win function.
    The function should return True if the player has won the game, and False otherwise.
    """
    board = ['X', 'X', 'X', '-', '-', '-', '-', '-', '-']
    assert check_win(board, 'X')
    assert not check_win(board, 'O')


def test_check_draw():
    """
    Test the check_draw function.
    The function should return True if the game is a draw, and False otherwise.
    """
    board = ['X', 'O', 'X', 'X', 'O', 'O', 'O', 'X', 'X']
    assert check_draw(board)

    board = ['X', 'O', 'X', 'X', '-', 'O', 'O', 'X', 'X']
    assert not check_draw(board)


def test_make_move():
    """
    Test the make_move function.
    The function should update the board state with the player's move.
    """
    board = ['-', '-', '-', '-', '-', '-', '-', '-', '-']
    make_move(board, 0, 'X')
    assert board[0] == 'X'
