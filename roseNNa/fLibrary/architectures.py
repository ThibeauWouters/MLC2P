import torch.nn as nn


# TODO provide documentation here
class FeedForwardNetwork(nn.Module):
    """
    Implements a simple feedforward neural network.
    """
    def __init__(self, h: list = [3, 600, 200, 1], activation_function = nn.Sigmoid, output_bias=True) -> None:
        """
        Initialize the neural network class.
        """
        # Call the super constructor first
        super(FeedForwardNetwork, self).__init__()

        # For convenience, save the sizes of the hidden layers as fields as well
        self.h = h

        # Define the layers:
        for i in range(len(self.h)-1):
            if i == len(self.h)-2:
                setattr(self, f"linear{i+1}", nn.Linear(self.h[i], self.h[i+1], bias=output_bias))
            else:
                setattr(self, f"linear{i+1}", nn.Linear(self.h[i], self.h[i+1]))
                setattr(self, f"activation{i+1}", activation_function())

    def forward(self, x):
        """
        Computes a forward step given the input x.
        :param x: Input for the neural network.
        :return: x: Output neural network
        """

        for i, module in enumerate(self.modules()):
            # The first module is the whole NNC2P object, continue
            if i == 0:
                continue
            x = module(x)

        return x