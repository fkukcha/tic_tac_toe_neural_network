# Generate training data
from api.build_model import build_model
from api.generate_training_data import generate_training_data


X_train, y_train = generate_training_data(10000)

# Build the model
model = build_model()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
