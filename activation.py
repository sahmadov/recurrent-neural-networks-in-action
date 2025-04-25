import numpy as np

class Activation:
    @staticmethod
    def softmax(logits):
        exp_values = np.exp(logits - np.max(logits))
        return exp_values / exp_values.sum(axis=0)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def derivative_tanh(x):
        return 1 - np.tanh(x) ** 2