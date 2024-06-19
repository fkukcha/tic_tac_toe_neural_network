def convert_board_to_numeric(board):
    """
    Converts a board to a numeric representation.
    :param board: The board.
    :return: A numeric representation of the board.
    """
    return [1 if cell == 'X' else -1 if cell == 'O' else 0 for cell in board]
