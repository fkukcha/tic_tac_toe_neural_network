import numpy as np
from matplotlib import pyplot as plt

from api.data_selection_preprocessing_analysis import X_test, y_test
from api.models_selection_training import model

"""
Make predictions on the test set
Reshape the test set to have the correct dimensions: (number of images, height, width, number of channels)
"""
y_pred = model.predict(X_test.reshape(-1, 28, 28, 1))
y_pred_classes = np.argmax(y_pred, axis=1)

"""
Find the errors
Errors occur when the predicted class is different from the true class
Show up to 10 errors, or less if fewer errors are found
"""
errors = np.where(y_pred_classes != y_test)[0]
num_errors_to_show = min(len(errors), 10)
plt.figure(figsize=(10, 5))
for i in range(num_errors_to_show):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[errors[i]], cmap='gray')
    plt.title(f"True: {y_test[errors[i]]}, Pred: {y_pred_classes[errors[i]]}")
plt.show()
