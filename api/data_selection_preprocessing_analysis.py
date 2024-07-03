import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_data(data_dir):
    images = []
    labels = []
    for label in ['X', 'O', 'Blank']:
        path = os.path.join(data_dir, label)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (28, 28))
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)


data_dir = '../tictactoe_images/train'
images, labels = load_data(data_dir)
images = images / 255.0

label_mapping = {'X': 0, 'O': 1, 'Blank': 2}
labels = np.array([label_mapping[label] for label in labels])

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Visualize the data
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(y_train[i])
plt.show()
