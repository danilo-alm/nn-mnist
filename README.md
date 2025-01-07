# Simple Neural Network in Python

This project demonstrates the implementation of a simple neural network from scratch using Python and `numpy` for mathematical operations. The neural network is trained on the MNIST dataset to classify handwritten digits (0-9).

No deep learning libraries like TensorFlow or PyTorch were used; the focus is on understanding the fundamentals of neural networks and backpropagation.

## Code Structure

### Neural Network Implementation
- **`NeuralNetwork` class:** Manages the forward propagation, backpropagation, parameter updates, and training iterations.
- **`Layer` class:** Represents a single layer of the network with activation functions, weights, and biases.

### Training Workflow
1. Initialize the neural network.
2. Add layers with specific configurations.
3. Train using gradient descent with a fixed or adaptive number of iterations.
4. Evaluate and predict using the trained model.


## How It Works

1. **Forward Propagation:** Computes the output of each layer by applying activation functions.
2. **Backpropagation:** Calculates gradients for weights and biases using the chain rule and updates them to minimize loss.
3. **Training Modes:**
   - **Fixed Iterations:** Runs for a predetermined number of iterations.
   - **Adaptive:** Stops training once accuracy stabilizes or fails to improve for a defined number of iterations.

## MNIST Dataset

The MNIST dataset consists of 70,000 images of handwritten digits (28x28 pixels) and their labels. It is a standard benchmark dataset for testing machine learning algorithms.
