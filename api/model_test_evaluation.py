from api.data_selection_preprocessing_analysis import X_test, y_test
from api.models_selection_training import model

"""
Evaluate the model on the test set
Reshape the test set to have the correct dimensions: (number of images, height, width, number of channels)
In this case, the number of channels is 1 because the images are grayscale
If the images were RGB, the number of channels would be 3
The number of images is -1 because we want to keep the number of images the same
X_test has the shape (number of images, height, width)
We need to add the number of channels to the shape
The new shape is (number of images, height, width, number of channels)
"""
test_loss, test_acc = model.evaluate(X_test.reshape(-1, 28, 28, 1), y_test)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_acc}")
