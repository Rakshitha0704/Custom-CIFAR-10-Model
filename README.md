# CIFAR-10 Image Classification using NumPy

This project implements a fully connected neural network (MLP) from scratch using NumPy to classify images from the CIFAR-10 dataset. It demonstrates how to build, train, and evaluate a neural network without using high-level frameworks like TensorFlow or PyTorch.


Features

- Fully connected neural network with:
  - One hidden layer (configurable size)
  - ReLU activation in hidden layer
  - Softmax output layer
- Forward and backward propagation implemented manually
- Mini-batch gradient descent training
- Model saving and loading using `pickle`
- Predicts and visualizes individual CIFAR-10 images with matplotlib
- Outputs classification report on test data


Tech Stack

- Python 3.x
- NumPy
- Keras (for loading CIFAR-10 dataset)
- Matplotlib
- Pickle (for saving/loading model parameters)


