import numpy as np

from api.convert_board_to_numeric import convert_board_to_numeric


def predict_move(board, model, optimal_only=False):
    """
    Predict the optimal move for the current board state.
    :param board: Current board state.
    :param model: Trained neural network model.
    :param optimal_only: If True, return only the predicted optimal move.
    :return: The predicted optimal move.
    """
    board_array = np.array([convert_board_to_numeric(board)], dtype=np.float32)
    prediction = model.predict(board_array)[0]
    if optimal_only:
        return np.argmax(prediction)
    return prediction
