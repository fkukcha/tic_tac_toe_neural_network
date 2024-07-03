import tensorflow as tf
import numpy as np
import cv2

from lime import lime_image
from matplotlib import pyplot as plt

from api.data_selection_preprocessing_analysis import X_test
from api.models_selection_training import model


def get_gradcam(model, img, layer_name='conv2d_2'):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([img]))
        loss = predictions[:, np.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)
    output = conv_outputs[0]
    grads = grads[0]
    weights = np.mean(grads, axis=(0, 1))
    cam = np.dot(output, weights)
    cam = cv2.resize(cam, (28, 28), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap


img = X_test[0]
heatmap = get_gradcam(model, img)
plt.imshow(img, cmap='gray')
plt.imshow(heatmap, alpha=0.5)
plt.show()

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(img, model.predict, top_labels=3, hide_color=0, num_samples=1000)
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
plt.imshow(temp)
plt.show()
