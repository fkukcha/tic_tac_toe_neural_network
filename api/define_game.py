# Function to check if the current player has won
def check_win(board, player):
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


# Function to check if the board is full (draw)
def check_draw(board):
    return all(cell != '-' for cell in board)


# Function to make a move
def make_move(board, position, player):
    board[position] = player
