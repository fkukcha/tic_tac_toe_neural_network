import numpy as np

from api.convert_board_to_numeric import convert_board_to_numeric


def predict_move(board, model):
    """
    Predict the optimal move for the current board state.
    :param board: Current board state.
    :param model: Trained neural network model.
    :return: Index of the optimal move.
    """
    board_array = np.array([convert_board_to_numeric(board)], dtype=np.float32)
    prediction = model.predict(board_array)[0]
    return np.argmax(prediction)
