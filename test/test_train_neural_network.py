import numpy as np

from api.build_model import build_model
from api.generate_training_data import generate_training_data


def test_train_neural_network():
    # Generate a small dataset for testing
    X_train, y_train = generate_training_data(1000)

    # Build the model
    model = build_model()

    # Train the model
    history = model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1)

    # Check if the model has been trained
    assert history.history['loss'][0] is not None
    assert history.history['accuracy'][0] is not None

    # Check if the model can make predictions
    sample_board = np.array([[1, -1, 0, 0, 1, 0, 0, -1, 1]], dtype=np.float32)
    prediction = model.predict(sample_board)
    assert prediction.shape == (1, 9)
