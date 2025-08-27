Dog vs. Cat Image Classification
This project demonstrates how to build and train a Convolutional Neural Network (CNN) to classify images of dogs and cats. The model is built using TensorFlow and Keras.

Project Overview
The goal of this project is to create a deep learning model that can accurately distinguish between images of dogs and cats. This is a classic computer vision problem that serves as a great introduction to building CNNs. The notebook covers all the essential steps, from data preparation and model creation to training and evaluation.

Dataset
The dataset used for this project is the "Dogs vs. Cats" dataset from Kaggle. It contains 25,000 images of dogs and cats, split into training and testing sets.

Training set: 20,000 images

Testing set: 5,000 images

You can download the dataset from here.

Model Architecture
The CNN model is built sequentially and consists of the following layers:

Conv2D Layer: 32 filters, kernel size of (3,3), ReLU activation. This layer is responsible for extracting features from the input images.

Batch Normalization: Normalizes the activations of the previous layer to improve training speed and stability.

MaxPooling2D: Downsamples the feature maps, reducing the spatial dimensions and computational complexity.

Conv2D Layer: 64 filters, kernel size of (3,3), ReLU activation.

Batch Normalization:

MaxPooling2D:

Conv2D Layer: 128 filters, kernel size of (3,3), ReLU activation.

Batch Normalization:

MaxPooling2D:

Flatten: Flattens the 3D output from the convolutional layers into a 1D vector.

Dense Layer: A fully connected layer with 128 neurons and ReLU activation.

Dense Layer: A fully connected layer with 64 neurons and ReLU activation.

Dropout Layer: A dropout layer with a rate of 0.1 to prevent overfitting.

Dense Layer (Output): A single neuron with a sigmoid activation function to output a probability between 0 and 1 (0 for cat, 1 for dog).

How to Run the Code
Prerequisites:

Python 3.x

Jupyter Notebook or Google Colab

Installation:
Install the necessary libraries:

pip install tensorflow opencv-python matplotlib

Set up Kaggle API:

Create a Kaggle account and download your kaggle.json API token.

Upload the kaggle.json file to your environment.

Download Dataset:
Run the notebook cells that download and extract the dataset from Kaggle.

Train the Model:
Execute the cells to define, compile, and train the CNN model on the training data.

Evaluate and Predict:

The notebook includes code to visualize the training/validation accuracy and loss.

You can use the trained model to predict on new images of dogs and cats.

Results
The model was trained for 15 epochs and achieved high accuracy on the training set and good performance on the validation set.

Training Accuracy: The model reaches over 99% accuracy on the training data.

Validation Accuracy: The validation accuracy fluctuates but generally stays around 80-85%, indicating some overfitting.

The training and validation accuracy/loss are plotted to visualize the model's performance over time.

Dependencies
TensorFlow

Keras

OpenCV

Matplotlib

NumPy


