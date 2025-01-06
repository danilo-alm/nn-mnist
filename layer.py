import numpy as np


class Layer:
    def __init__(self, n_neurons, n_input, activation, weights=None, bias=None):
        self.weights = weights or np.random.rand(n_neurons, n_input) - 0.5
        self.bias = bias or np.random.rand(n_neurons, 1) - 0.5
        self.activation = activation
        self.last_activation = None
        self.pre_activation_values = None

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
        
    def ReLU(self, x):
        return np.maximum(x, 0)
    
    def derivative_ReLU(self, x):
        return np.where(x <= 0, 0, 1)
    
    def apply_activation_derivative(self):
        if self.activation == 'relu':
            return self.derivative_ReLU(self.pre_activation_values)
        raise NotImplementedError('Activation function not implemented')

    def _apply_activation(self, r):
        if self.activation == 'softmax':
            return self.softmax(r)
        elif self.activation == 'relu':
            return self.ReLU(r)
        raise NotImplementedError('Activation function not implemented')

    def activate(self, x):
        self.pre_activation_values = self.weights.dot(x) + self.bias
        self.last_activation = self._apply_activation(self.pre_activation_values)
        return self.last_activation