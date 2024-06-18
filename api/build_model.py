import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


def build_model():
    model = Sequential([
        Flatten(input_shape=(9,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(9, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
