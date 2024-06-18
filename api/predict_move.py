import numpy as np


# Function to predict the optimal move
def predict_move(board, model):
    board_array = np.array([board])
    prediction = model.predict(board_array)[0]
    return np.argmax(prediction)
