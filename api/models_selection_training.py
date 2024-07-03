from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

from api.data_selection_preprocessing_analysis import X_train, y_train

"""
Create a simple CNN model
The input shape is (28, 28, 1) because the images are grayscale
The output layer has 3 units because there are 3 classes: X, O, and Blank
The activation function for the output layer is softmax because this is a multi-class classification problem
relu is used as the activation function for the hidden layers
Convolutional layers are used to extract features from the images
Max pooling layers are used to reduce the spatial dimensions of the feature maps
Flatten is used to convert the 2D feature maps to a 1D vector
Dense layers are used to make the final predictions
"""
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])

"""
The optimizer is adam
The loss function is sparse_categorical_crossentropy because the labels are integers
The metrics are accuracy
"""
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

"""
Train the model
The input shape is (number of images, height, width, number of channels)
In this case, the number of channels is 1 because the images are grayscale
If the images were RGB, the number of channels would be 3
The number of images is -1 because we want to keep the number of images the same
X_train has the shape (number of images, height, width)
We need to add the number of channels to the shape
The new shape is (number of images, height, width, number of channels)
"""
history = model.fit(X_train.reshape(-1, 28, 28, 1), y_train, epochs=10, validation_split=0.2)
