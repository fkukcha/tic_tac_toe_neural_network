def get_human_move(board):
    """
    Gets the human player's move.
    :param board: Current board state.
    :return: The human player's move.
    """
    while True:
        try:
            move = int(input("Enter your move (1-9): ")) - 1  # Convert input to 0-based index
            if move < 0 or move > 8 or board[move] != '-':
                raise ValueError
            return move
        except ValueError:
            print("Invalid move. Please enter a number between 1 and 9 for an empty cell.")
