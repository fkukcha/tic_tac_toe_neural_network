# Tic-Tac-Toe Image Classification

## Development Environment and Branch Information

- **IDE Used**: PyCharm.

- **Git Branch**: Please use the `master` branch for this project. All the latest stable versions of the project files are maintained in the `master` branch.


## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Setting up the Environment](#setting-up-the-environment)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Selection, Preprocessing, and Analysis](#data-selection-preprocessing-and-analysis)
  - [Model Selection and Training](#model-selection-and-training)
  - [Optimization](#optimization)
  - [Error Analysis](#error-analysis)
  - [Model Test Evaluation](#model-test-evaluation)
  - [Comparing Advanced Models](#comparing-advanced-models)
  - [Explainable Techniques and Tools](#explainable-techniques-and-tools)

## Project Overview
This project is focused on classifying Tic-Tac-Toe game images using machine learning techniques. The goal is to classify images into three categories: 'X', 'O', and 'Blank'. The project includes data loading, preprocessing, model selection, training, optimization, error analysis, and explainability techniques.

## Requirements
- Python 3.x
- tensorflow
- numpy
- matplotlib
- scikit-learn
- lime
- scikeras

Install the required packages using `pip`:
```bash
pip install -r requirements.txt
```

## Setting up the Environment
1. We start by installing the package- and environment manager conda
2. Therefore, download Miniconda here:
> https://docs.conda.io/en/latest/miniconda.html
> 
> Available for Windows, MacOS, and Linux
3. Install Miniconda
> When prompted, add Miniconda3 to PATH environment
> 
> Otherwise, you won‘t be able to use conda from your terminal
4. Testing conda
> If you added Miniconda3 to your PATH variable, load your favorite
terminal and execute the following command:
`conda --version`
5. Updating Conda
> Update conda, using the following command:
> 
> conda update -n base -c defaults conda
6. Create your virtual environment
> 1. Create a new virtual environment:
> > conda activate base
> 
> > conda create --name `<env name>`
> 
> > or
> > conda create --prefix `<env name>` `<more options>`
> 2. Activate your virtual environment:
> > conda activate `<env name>` or conda activate `./<env name>`
> 3. Install the required packages:
> > conda install `<package name>`
> > pip install `<package name>`
> 4. To deactivate your virtual environment:
> > conda deactivate

## Directory Structure
```plaintext
.
├── api
│   ├── data_selection_preprocessing_analysis.py
│   ├── model_selection_training.py
│   ├── optimization.py
│   ├── error_analysis.py
│   ├── model_test_evaluation.py
│   ├── comparing_advanced_models.py
│   └── explainable_techniques_tools.py
├── tictactoe_images
│   ├── test
│   │   ├── Blank
│   │   ├── O
│   │   └── X
│   └── train
│       ├── Blank
│       ├── O
│       └── X
├── README.md
└── requirements.txt
```

## Data Selection, Preprocessing, and Analysis

The `api/data_selection_preprocessing_analysis.py` file is responsible for the initial stages of the machine learning pipeline, focusing on data selection, preprocessing, and analysis for the Tic-Tac-Toe image classification project. This script performs several key tasks to prepare the data for model training and evaluation.

### Key Functions

- `load_data(data_dir: str) -> (np.array, np.array)`: This function loads the images and their corresponding labels from the specified directory. It processes each image by converting it to grayscale, resizing it to a uniform dimension (28x28 pixels), and appending it to an array. The labels are also stored in an array. The function returns two arrays: one for the images and one for the labels.

### Workflow

1. **Data Loading**: The script starts by defining the `load_data` function, which takes the path to the data directory as input. It iterates over each subdirectory (representing a class label) and loads the images into a NumPy array. Each image is converted to grayscale and resized to ensure uniformity.

2. **Data Normalization**: After loading, the images are normalized by dividing each pixel value by 255.0, scaling the pixel values to a range of 0 to 1. This normalization step is crucial for the convergence of the neural network during training.

3. **Label Encoding**: The labels (‘X’, ‘O’, and ‘Blank’) are converted to numerical values (0, 1, and 2, respectively) to facilitate processing by the machine learning model.

4. **Train-Test Split**: The dataset is split into training and testing sets, with 80% of the data used for training and 20% reserved for testing. This split is performed to evaluate the model's performance on unseen data.

5. **Data Visualization**: Finally, the script visualizes a subset of the training images along with their labels. This step is helpful for verifying the data loading and preprocessing steps.

## Model Selection and Training

The `api/models_selection_training.py` file is central to the machine learning aspect of the Tic-Tac-Toe image classification project. It outlines the process of selecting and training the convolutional neural network (CNN) model that classifies Tic-Tac-Toe game images into three categories: 'X', 'O', and 'Blank'.

### Overview

This script defines a simple yet effective CNN architecture tailored for the task of image classification. The model consists of convolutional layers for feature extraction, max pooling layers for dimensionality reduction, and dense layers for prediction.

### Model Architecture

- **Input Layer**: Accepts grayscale images with a shape of (28, 28, 1).
- **Convolutional Layers**: Two convolutional layers with 32 and 64 filters, respectively, each followed by a max pooling layer. These layers are responsible for extracting features from the input images.
- **Flatten Layer**: Converts the 2D feature maps into a 1D vector to be used in the dense layers.
- **Dense Layers**: A dense layer with 128 units for further processing of features, followed by the output layer with 3 units corresponding to the three classes ('X', 'O', and 'Blank'). The softmax activation function is used in the output layer for multi-class classification.

### Training Process

- **Optimizer**: Adam, known for its efficiency in training deep learning models.
- **Loss Function**: Sparse categorical crossentropy, suitable for multi-class classification problems where the classes are represented as integers.
- **Metrics**: Accuracy, to measure the performance of the model during training and validation.

## Optimization Process

The `api/optimization.py` file is dedicated to optimizing the hyperparameters of the convolutional neural network (CNN) model used in the Tic-Tac-Toe image classification project. This script employs grid search to find the best optimizer for the model, enhancing its performance on the classification task.

### Overview

Hyperparameter optimization is a crucial step in machine learning to improve model performance without changing the model architecture. This script focuses on optimizing the model's optimizer by comparing different options.

### Key Components

- **Model Creation Function**: A function `create_model` that defines the CNN model architecture and compiles it with a given optimizer. This allows for easy adjustment and testing of different optimizers during the grid search process.

- **Grid Search**: Utilizes `GridSearchCV` from `scikit-learn` to systematically work through multiple combinations of parameter options, cross-validating as it goes to determine which parameters yield the best model performance.

### Optimizers Tested

- **Adam**: A popular optimizer that combines the best properties of the AdaGrad and RMSProp algorithms to provide an optimization algorithm that can handle sparse gradients on noisy problems.
- **RMSprop**: An optimizer that utilizes the magnitude of recent gradients to normalize the gradients. It's very effective for recurrent neural networks.

### Implementation Details

- The CNN model is wrapped in a `KerasClassifier` to make it compatible with `scikit-learn`'s `GridSearchCV`.
- The grid search explores the model performance over two optimizers: `adam` and `rmsprop`.
- The process involves reshaping the training data to fit the model's input requirements, specifying the number of jobs to use all available processors for parallel computation, and setting the cross-validation folds to 3.

## Error Analysis

The `api/error_analysis.py` file plays a crucial role in evaluating the performance of the convolutional neural network (CNN) model used in the Tic-Tac-Toe image classification project. This script is designed to identify and visualize the errors made by the model on the test dataset, providing insights into the model's prediction capabilities and areas for improvement.

### Key Components

- **Predictions on Test Set**: The script starts by making predictions on the test set. The test images are reshaped to match the input dimensions expected by the model, ensuring accurate prediction results.

- **Identification of Errors**: It identifies errors by comparing the predicted classes with the true classes from the test set. An error is recorded when there is a mismatch between the predicted and true class.

- **Visualization of Errors**: The script visualizes up to 10 errors (or fewer if there are not many errors) by displaying the misclassified images alongside their true and predicted labels. This visualization helps in understanding the nature of the errors, such as whether they are due to ambiguous images, preprocessing issues, or limitations of the model.

## Model Testing and Evaluation

The `api/model_test_evaluation.py` file is crucial for assessing the performance of the convolutional neural network (CNN) model developed for the Tic-Tac-Toe image classification project. This script evaluates the model's accuracy and loss on the test dataset, providing a quantitative measure of its ability to classify unseen images correctly.

### Key Components

- **Model Evaluation**: The script uses the `evaluate` method of the model to compute the loss and accuracy on the test set. This method compares the model's predictions against the true labels to determine its performance.

- **Data Reshaping**: Before evaluation, the test images are reshaped to match the input dimensions expected by the model. For grayscale images, this involves adding an additional dimension to represent the single color channel.

## Advanced Model Comparison

The `api/compare_advanced_models.py` file is designed to explore the use of more advanced neural network architectures for the Tic-Tac-Toe image classification project. Specifically, it implements a model based on the ResNet50 architecture, known for its deep structure and ability to learn from a significant amount of data.

### Overview

This script demonstrates how to adapt the ResNet50 model, pre-trained on the ImageNet dataset, for the task of classifying Tic-Tac-Toe game images into three categories: 'X', 'O', and 'Blank'. It includes steps for resizing images to fit the input requirements of ResNet50, modifying the model to suit our specific classification task, and training the model on the project's dataset.

### Key Components

- **ResNet50 Base Model**: Utilizes the ResNet50 model from Keras applications, pre-trained on ImageNet. The model is adapted to accept input images of size 32x32 pixels.
- **Global Average Pooling**: Applies a GlobalAveragePooling2D layer to reduce the spatial dimensions of the feature maps from the ResNet50 base model, making it suitable for classification.
- **Dense Layers**: Adds a dense layer with 128 units and ReLU activation for further feature processing, followed by the output layer with 3 units and softmax activation to classify the images into the three categories.

### Model Adaptation

- **Input Shape Adjustment**: The images are resized to 32x32 pixels to match the input shape requirement of ResNet50, which is modified to accept three-channel (RGB) images of this size.
- **Output Layer Customization**: The final layers of the model are customized to reflect the project's specific classification needs, with three output classes and softmax activation.

### Training Process

- **Freezing Base Model Layers**: The layers of the ResNet50 base model are frozen to prevent their weights from being updated during training, allowing the model to leverage pre-learned features.
- **Compilation**: The model is compiled with the Adam optimizer and sparse categorical crossentropy loss function, suitable for multi-class classification.
- **Model Training**: The model is trained on the resized training images with labels, using a validation split to monitor performance and avoid overfitting.

## Explainable AI Techniques and Tools

The `api/explainable_techniques_tools.py` file incorporates advanced techniques for understanding and interpreting the decisions made by the convolutional neural network (CNN) model used in the Tic-Tac-Toe image classification project. This script utilizes Gradient-weighted Class Activation Mapping (Grad-CAM) and Local Interpretable Model-agnostic Explanations (LIME) to visualize the model's reasoning behind its predictions.

### Overview

Explainable AI (XAI) aims to make the outcomes of AI models more understandable to humans. This script demonstrates the application of two popular XAI techniques, Grad-CAM and LIME, which provide insights into the model's decision-making process by highlighting the important regions in the input images that influence the model's predictions.

### Key Components

- **Grad-CAM**: Generates heatmaps for given images based on the gradients of the target concept (class being predicted) flowing into the final convolutional layer of the CNN model. This helps in visualizing which parts of the image are important for the model's decision.

- **LIME**: Provides local interpretable explanations for model predictions. It perturbs the input image and observes the impact on the output to identify regions that significantly influence the prediction. This method is model-agnostic, meaning it can be used with any model.

### Implementation Details

- **Grad-CAM Implementation**: The `get_gradcam` function generates a heatmap for a specified image and layer. It involves creating a model that outputs the selected layer's activations and the model's predictions, calculating the gradients of the target class with respect to the activations, and using these gradients to produce a weighted heatmap of the important regions.

- **LIME Implementation**: Utilizes the `LimeImageExplainer` to create an explanation for a given image. The explanation identifies the top contributing features (pixels or regions) for the model's prediction and visualizes them, helping to understand why the model made a certain decision.
