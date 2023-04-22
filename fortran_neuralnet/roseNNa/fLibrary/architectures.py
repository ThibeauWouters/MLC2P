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


class Net(nn.Module):
    """
    Implements a simple feedforward neural network.
    NOTE - this is the same as above but just slightly different implementation of hidden layer input argument.
    TODO - this makes it cumbersome, have to adapt this later on. 
    """

    def __init__(self, nb_of_inputs: int = 3, nb_of_outputs: int = 1, h: list = [600, 200], reg: bool = False,
                 activation_function=nn.Sigmoid) -> None:
        """
        Initialize the neural network class.
        """
        # Call the super constructor first
        super(Net, self).__init__()

        # For convenience, save the sizes of the hidden layers as fields as well
        self.h = h
        # Add visible layers as well: input is 3D and output is 1D
        self.h_augmented = [nb_of_inputs] + h + [nb_of_outputs]

        # Add field to specify whether or not we do regularization
        self.regularization = reg

        # Define the layers:
        for i in range(len(self.h_augmented) - 1):
            if i == len(self.h_augmented) - 2:
                setattr(self, f"linear{i + 1}", nn.Linear(self.h_augmented[i], self.h_augmented[i + 1], bias=False))
            else:
                setattr(self, f"linear{i + 1}", nn.Linear(self.h_augmented[i], self.h_augmented[i + 1]))
                setattr(self, f"activation{i + 1}", activation_function())

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