import numpy as np
import copy


class NeuralNetwork:
    def __init__(self):
        self._layers = []

    def _one_hot_encode(self, y):
        y_encoded = np.zeros((y.size, y.max() + 1))
        y_encoded[np.arange(y.size), y] = 1
        return y_encoded.T

    def _forward_prop(self, X):
        for layer in self._layers:
            X = layer.activate(X)
        return X
    
    def _back_prop(self, X, y):
        m = y.size
        one_hot_y = self._one_hot_encode(y)

        predictions = self._layers[1].last_activation

        dZ2 = predictions - one_hot_y
        dW2 = 1 / m * dZ2.dot(self._layers[0].last_activation.T)
        db2 = 1 / m * np.sum(dZ2)
        
        dZ1 = self._layers[1].weights.T.dot(dZ2) * self._layers[0].apply_activation_derivative()
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1)

        return dW1, db1, dW2, db2

    def _update_params(self, dW1, db1, dW2, db2, learning_rate):
        l1 = self._layers[0]
        l2 = self._layers[1]

        l1.weights -= learning_rate * dW1
        l1.bias -= learning_rate * db1
        
        l2.weights -= learning_rate * dW2
        l2.bias -= learning_rate * db2

    def _get_accuracy(self, y):
        predictions = self._layers[1].last_activation
        return np.sum(predictions.argmax(axis=0) == y) / y.size

    def _perform_iteration(self, X, y, i, lr):
        self._forward_prop(X)
        self._update_params(*self._back_prop(X, y), lr)
        if i % 10 == 0:
            self._log_iteration_info(i, y)

    def _log_iteration_info(self, i, y):
        print(f'Iteration {i}')
        print(f'Accuracy: {self._get_accuracy(y)}')
        print('---')

    def _fixed_iterations_mode(self, X, y, iterations, lr):
        for i in range(iterations):
            self._perform_iteration(X, y, i, lr)

    def _adaptive_iterations_mode(self, X, y, iterations, lr):
        i = 0
        did_not_improve = 0
        best_accuracy = 0
        best_params = None

        while True:
            self._perform_iteration(X, y, i, lr)
            i += 1

            accuracy = self._get_accuracy(y)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = copy.deepcopy(self._layers)
                did_not_improve = 0
            else:
                did_not_improve += 1

            if did_not_improve >= abs(iterations):
                self._log_final_results(best_accuracy, i)
                break

        self._layers = best_params

    def _log_final_results(self, best_accuracy, total_iterations):
        print('---')
        print(f'Best accuracy: {best_accuracy}')
        print(f'{total_iterations} iterations')
        print('---')
    
    def add_layer(self, layer):
        if len(self._layers) >= 2:
            raise ValueError('This implementation only supports two layers')
        self._layers.append(layer)

    def gradient_descent(self, X, y, iterations, lr):
        if len(self._layers) < 2:
            raise ValueError('This implementation only supports two layers')

        if iterations >= 0:
            self._fixed_iterations_mode(X, y, iterations, lr)
        else:
            self._adaptive_iterations_mode(X, y, iterations, lr)

    def predict(self, X):
        return np.argmax(self._forward_prop(X), 0)