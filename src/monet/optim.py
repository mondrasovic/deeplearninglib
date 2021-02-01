"""
We use an optimizer to adjust the parameters of our network based 
on the gradients computed during backpropagation.
"""

import abc

from monet.nn import NeuralNet


class Optimizer(abc.ABC):

    @abc.abstractmethod
    def step(self, net: NeuralNet) -> None:
        pass


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
    
    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad
