import numpy as np
from api.generate_training_data import generate_training_data


def test_generate_training_data():
    """
    Test if the generate_training_data function returns the correct data type and shape.
    The function should return a tuple of two numpy arrays, X and y, where X is the training data and y is the labels.
    The training data should have shape (n, 9) and the labels should have shape (n, 9), where n is the number of
    training examples.
    """
    X, y = generate_training_data(10)
    assert len(X) == len(y)
    assert len(X) > 0
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[1] == 9  # Each board state should have 9 elements
    assert y.shape[1] == 9  # Each label should have 9 elements

    # Check if the values in X are valid (1 for 'X', -1 for 'O', 0 for empty)
    for board in X:
        assert all(cell in [1, -1, 0] for cell in board)
