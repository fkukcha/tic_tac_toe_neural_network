from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input

from api.data_selection_preprocessing_analysis import X_train, y_train


def create_model(optimizer='adam'):
    """
    Create a simple CNN model
    :param optimizer: str, optimizer to use
    :return: Sequential, CNN model
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
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


"""
Grid search for the optimizer
The optimizer is a hyperparameter that can be tuned
We will search over two optimizers: adam and rmsprop
The model is created using the create_model function
The model is wrapped in a KerasClassifier to be used with scikit-learn
The grid search is performed using GridSearchCV
The number of jobs is set to -1 to use all available processors
The cross-validation is set to 3 folds
The input shape is (number of images, height, width, number of channels)
In this case, the number of channels is 1 because the images are grayscale
If the images were RGB, the number of channels would be 3
The number of images is -1 because we want to keep the number of images the same
X_train has the shape (number of images, height, width)
We need to add the number of channels to the shape
The new shape is (number of images, height, width, number of channels)
"""
model = KerasClassifier(model=create_model)
optimizers = ['adam', 'rmsprop']
param_grid = dict(optimizer=optimizers)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train.reshape(-1, 28, 28, 1), y_train)
