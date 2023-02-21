"""Python script for code related to neural networks used in the C2P conversion."""

import physics
import data
from torch import nn


# Implement neural network
class NeuralNetwork(nn.Module):
    """
    Implements a two-layered neural network for the C2P conversion. Note that hence the number of layers is fixed
    for this NN subclass! The activation functions are sigmoids.
    """
    def __init__(self, h1: int, h2: int):
        """
        Initialize the neural network class.
        :param h1: Size (number of neurons) of the first hidden layer.
        :param h2: Size (number of neurons) of the second hidden layer.
        """
        # Call the super constructor first
        super(NeuralNetwork, self).__init__()
        # TODO - remove this?
        #self.flatten = nn.Flatten()

        # Network uses sigmoid activation functions. Input has size 3 (D, S, tau) and returns the pressure.
        self.stack = nn.Sequential(
            nn.Linear(3, h1),
            nn.Sigmoid(),
            nn.Linear(h1, h2),
            nn.Sigmoid(),
            nn.Linear(h2, 1)
        )

    def forward(self, x):
        """
        Computes a forward step given the input x.
        :param x: Input for the neural network.
        :return:
        """
        # TODO - No flatten needed, as our input and output are 1D?
        # x = self.flatten(x)
        logits = self.stack(x)
        return logits


def compute_loss(pred, y, loss_fn, model, regularization=False, lambdaa=0.001):
    """
    Computes the loss function, possibly with a regularization term with coefficient lambda.
    Note: Replace abs() with pow(2.0) for L2 regularization

    :param pred: The value predicted by the neural network.
    :param y: The real value of the output.
    :param loss_fn: The base function used to compute the loss function, implemented in optimizer.
    :param regularization: Boolean indicating whether or not we use regularization.
    :param lambdaa: The coefficient in front of the regularization term.
    :return:
    """

    # Use the base loss function first
    loss = loss_fn(pred, y)

    # If we use regularization, add the regularization term
    if regularization:
        # Compute the norm
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        # Scale according to the specified coefficient
        loss += lambdaa * l1_norm

    return loss
