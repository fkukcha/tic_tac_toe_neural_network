import tensorflow as tf
from keras.src.applications.resnet import ResNet50

from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense

from api.data_selection_preprocessing_analysis import X_train, y_train

# Resize images to 32x32
X_train_resized = tf.image.resize(X_train, [32, 32])

"""
Create a more advanced model using ResNet50
The input shape is (32, 32, 3) because the images are resized to 32x32 and are RGB
The output layer has 3 units because there are 3 classes: X, O, and Blank
The activation function for the output layer is softmax because this is a multi-class classification problem
relu is used as the activation function for the hidden layers
GlobalAveragePooling2D is used to reduce the spatial dimensions of the feature maps
Dense layers are used to make the final predictions
"""
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)
advanced_model = tf.keras.Model(inputs=base_model.input, outputs=predictions)


# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
advanced_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
advanced_model.summary()

# Train the model
history = advanced_model.fit(X_train_resized, y_train, epochs=10, validation_split=0.2)
