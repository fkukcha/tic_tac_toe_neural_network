from api.data_selection_preprocessing_analysis import X_test, y_test
from api.models_selection_training import model

test_loss, test_acc = model.evaluate(X_test.reshape(-1, 28, 28, 1), y_test)
print(f"Test accuracy: {test_acc}")
