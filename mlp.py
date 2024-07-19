import numpy as np

class LinearLayer:
    def __init__(self, input_D, output_D):
        self.params = {
            'W': np.random.randn(input_D, output_D) * np.sqrt(2. / input_D),  # He initialization
            'b': np.zeros(output_D)
        }
        self.gradient = {
            'W': np.zeros((input_D, output_D)),
            'b': np.zeros(output_D)
        }

    def forward(self, X):
        forward_output = np.dot(X, self.params['W']) + self.params["b"]
        return forward_output

    def backward(self, X, grad):
        self.gradient['W'] = np.dot(X.T, grad)
        self.gradient['b'] = np.sum(grad, axis=0)
        backward_output = np.dot(grad, self.params["W"].T)
        return backward_output

class ReLU:

    def __init__(self):
        self.mask = None

    def forward(self, X):
        forward_output = np.maximum(0, X)
        self.mask = (forward_output > 0)
        return forward_output

    def backward(self, X, grad):
        backward_output = grad * self.mask
        return backward_output

class LeakyReLU:

    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.mask = None

    def forward(self, X):
        forward_output = np.where(X > 0, X, self.alpha * X)
        self.mask = (X > 0)
        return forward_output

    def backward(self, X, grad):
        backward_output = grad * np.where(X > 0, 1, self.alpha)
        return backward_output