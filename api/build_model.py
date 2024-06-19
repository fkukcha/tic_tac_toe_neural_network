from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


def build_model():
    """
    This function is used to build the model for the neural network.
    The model consists of 3 layers: 1 input layer, 2 hidden layers, and 1 output layer.
    The input layer has 9 neurons (one for each cell in the board).
    The hidden layers have 128 and 64 neurons respectively.
    The output layer has 9 neurons (one for each possible move).
    The activation function used in the hidden layers is ReLU.
    The activation function used in the output layer is softmax.
    :return: model: The compiled neural network model.
    """
    model = Sequential([
        Flatten(input_shape=(9,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(9, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
