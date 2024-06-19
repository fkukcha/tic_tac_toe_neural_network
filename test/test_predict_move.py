import numpy as np
from api.build_model import build_model
from api.predict_move import predict_move


def test_predict_move():
    """
    Test the predict_move function.
    The function should return a valid move given a board state.
    """
    model = build_model()
    # Mock training data for the model
    X_train = np.array([[1, -1, 0, 0, 1, 0, 0, -1, 1]], dtype=np.float32)
    y_train = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    model.fit(X_train, y_train, epochs=1, verbose=0)

    board = ['X', 'O', '-', '-', 'X', '-', '-', 'O', 'X']
    move = predict_move(board, model)
    assert move in range(9)  # Ensure the move is a valid board index
