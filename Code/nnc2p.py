"""Python script for code related to neural networks used in the C2P conversion."""

from typing import Callable
import numpy as np
import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from data import CustomDataset


#########################
# NETWORK ARCHITECTURES #
#########################


class NeuralNetwork(nn.Module):
    """
    Implements a two-layered neural network for the C2P conversion. Note that hence the number of layers is fixed
    for this NN subclass! The activation functions are sigmoids.
    """
    def __init__(self, name: str = "network", h1: int = 600, h2: int = 200, reg: bool = False) -> None:
        """
        Initialize the neural network class.
        :param name: String that names this network, in order to recognize it later on.
        :param h1: Size (number of neurons) of the first hidden layer.
        :param h2: Size (number of neurons) of the second hidden layer.
        """
        # Call the super constructor first
        super(NeuralNetwork, self).__init__()

        # Add field to specify whether or not we do regularization
        self.regularization = reg
        self.name = name

        # Define the weights:
        self.linear1 = nn.Linear(3, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, 1)

        # Network uses sigmoid activation functions. Input has size 3 (D, S, tau) and returns the pressure.
        # previous code:
        # self.stack = nn.Sequential(
        #     nn.Linear(3, h1),
        #     nn.Sigmoid(),
        #     nn.Linear(h1, h2),
        #     nn.Sigmoid(),
        #     nn.Linear(h2, 1)
        # )

    # TODO - what is input type here?
    def forward(self, x):
        """
        Computes a forward step given the input x.
        :param x: Input for the neural network.
        :return: x: Output neural network
        """

        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
        return x

    def load_parameters(self, state_dict):
        """
        Loads the parameters, as saved before, into this network's layers. Note: assumes 2-layer network, same size.
        Explanation: we changed the details of the neural network architecture definition to have named modules in order
        to better control the pruning methods.

        :param state_dict: A state_dict containing parameters.
        :return: None
        """

        # First layer
        stack_0_weight = state_dict['stack.0.weight']
        stack_0_bias   = state_dict['stack.0.bias']
        self.linear1.weight = nn.parameter.Parameter(stack_0_weight)
        self.linear1.bias  = nn.parameter.Parameter(stack_0_bias)

        # Second layer
        stack_2_weight = state_dict['stack.2.weight']
        stack_2_bias = state_dict['stack.2.bias']
        self.linear2.weight = nn.parameter.Parameter(stack_2_weight)
        self.linear2.bias = nn.parameter.Parameter(stack_2_bias)

        # Third layer
        stack_4_weight = state_dict['stack.4.weight']
        stack_4_bias = state_dict['stack.4.bias']
        self.linear3.weight = nn.parameter.Parameter(stack_4_weight)
        self.linear3.bias = nn.parameter.Parameter(stack_4_bias)


############################
# GENERAL TRAINING ASPECTS #
############################


def compute_loss(prediction, y, loss_fn, model, lambdaa=0.001) -> torch.tensor:
    """
    Computes the loss function, possibly with a regularization term with coefficient lambda.
    Note: Replace abs() with pow(2.0) for L2 regularization

    :param prediction: The value predicted by the neural network.
    :param y: The real value of the output.
    :param loss_fn: The base function used to compute the loss function, implemented in optimizer.
    :param model: The neural network.
    :param lambdaa: The coefficient in front of the regularization term.
    :return: loss: The loss for a single instance.
    """

    # Use the base loss function first
    loss = loss_fn(prediction, y)

    # If we use regularization, add the regularization term
    if model.regularization:
        # Compute the norm
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        # Scale according to the specified coefficient
        loss += lambdaa * l1_norm

    return loss


def train_loop(dataloader: DataLoader, model: NeuralNetwork, loss_fn: Callable, optimizer: torch.optim.Adam,
               report_progress: bool = False) -> None:
    """
    Does one epoch of the training loop.

    :param dataloader: Torch DataLoader object, containing training data.
    :param model: The neural network.
    :param loss_fn: The loss function on which the neural network is trained.
    :param optimizer: The optimization algorithm used.
    :param report_progress: Boolean indicating whether or not we want to print the progress of training.
    :return: None
    """
    # TODO - what's the issue here?
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        prediction = model(X)
        loss = compute_loss(prediction, y, loss_fn, model)

        # Do the backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # If we want to report progress during training (not recommended - obstructs view)
        if report_progress:
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader: DataLoader, model: NeuralNetwork, loss_fn: Callable) -> float:
    """
    The testing loop for a single epoch.

    :param dataloader: Torch DataLoader object.
    :param model: The neural network.
    :param loss_fn: The loss function used during training.
    :return: test_loss: Loss computed on the test data.
    """

    # Get the number of batches
    num_batches = len(dataloader)
    test_loss = 0

    # Predict and compute losses
    with torch.no_grad():
        for X, y in dataloader:
            prediction = model(X)
            test_loss += compute_loss(prediction, y, loss_fn, model).item()

    # Return the average of the loss over the number of batches
    return test_loss / num_batches


#################
# TRAINER CLASS #
#################


class Trainer:

    def __init__(self, train_dataloader: DataLoader, test_dataloader: DataLoader, model: NeuralNetwork,
                 loss_fn: Callable, optimizer: torch.optim.Adam, train_losses=None, test_losses: list = None,
                 adaptation_indices: list = None) -> None:
        """
        Initializes a Trainer object. This object brings together the neural network architecture defined above,
        the optimizer used, the dataloaders used while training, and simplifies the training process for the
        notebooks.

        :param model: The neural network.
        :param optimizer: The optimizer used during training.
        :param train_losses: List that contains the train losses during training, for plotting purposes.
        :param test_losses: List that contains the test losses during training, for plotting purposes.
        :param adaptation_indices: Indices at which the learning rate gets adapted, for plotting purposes.
        """
        # Save the model and optimizer as fields
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        # For list arguments, make sure that empty list is initialized as default
        if train_losses is None:
            train_losses = []
        self.train_losses = train_losses
        if test_losses is None:
            test_losses = []
        self.test_losses = test_losses
        if adaptation_indices is None:
            adaptation_indices = []
        self.adaptation_indices = adaptation_indices
        # Save a boolean indicating whether or not we have trained this network before (ie, train_losses empty or not)
        if len(self.train_losses) == 0:
            self.trained = True
        else:
            self.trained = False

    # def __init__(self, trainer: Trainer) -> None:
    #     # Save the model and optimizer as fields
    #     self.model = trainer.model
    #     self.optimizer = trainer.optimizer
    #     self.trained = trainer.trained
    #     self.train_losses = trainer.train_losses
    #     self.test_losses = trainer.test_losses
    #     self.adaptation_indices = trainer.daptation_indices
    #     self.trained = trainer.trained

    def train(self, adaptation_threshold=0.9995, adaptation_multiplier=0.5, number_of_epochs: int = 500):
        """

        :param adaptation_threshold:
        :param adaptation_multiplier:
        :param number_of_epochs:
        :return:
        """

        print("Training the model . . .")

        # Initialize a few auxiliary variables
        epoch_counter = 0  # epoch counter for this training session
        total_epoch_counter = len(self.train_losses) + 1  # overall counter of training epochs for this network

        # The counter makes sure we do not update the learning rate too often
        # if the network was not trained yet, first 5 epochs we don't change the learning rate
        if not self.trained:
            counter = -5
        else:
            counter = 0

        # Keep on continuing the training until we hit max number of epochs
        while epoch_counter < number_of_epochs:
            print(f"\n Epoch {epoch_counter} \n --------------")
            train_loop(self.train_dataloader, self.model, self.loss_fn, self.optimizer)
            # Test on the training data
            average_train_loss = test_loop(self.train_dataloader, self.model, self.loss_fn)
            self.train_losses.append(average_train_loss)
            # Test on testing data
            average_test_loss = test_loop(self.test_dataloader, self.model, self.loss_fn)
            self.test_losses.append(average_test_loss)

            # Update the learning rate - see Appendix B of the paper
            # only check if update needed after 10 new epochs
            if counter >= 10:
                current = np.min(self.train_losses[-5:])
                previous = np.min(self.train_losses[-10:-5])

                # If we did not improve the test loss sufficiently, going to adapt LR
                if current / previous >= adaptation_threshold:
                    # Reset counter (note: will increment later, so set to -1 st it becomes 0)
                    counter = -1
                    old_learning_rate = self.optimizer.param_groups[-1]['lr']
                    learning_rate = adaptation_multiplier * old_learning_rate
                    print(f"Adapting learning rate to {learning_rate}")
                    # Change optimizer
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
                    # Add the epoch time for plotting later on
                    self.adaptation_indices.append(epoch_counter)

            # Report progress:
            # print(f"Average loss of: {average_test_loss} for test data")
            print(f"Average loss of: {average_train_loss} for train data")

            # Another epoch passed - increment counter
            counter += 1
            epoch_counter += 1

        print("Done!")

###########################
# PERFORMANCE OF NETWORKS #
###########################


# TODO - type hinting?
def l1_norm(predictions, y):
    """
    Computes the L1 norm between predictions made by the neural network and the actual values.
    :param predictions: Predictions made by the neural network architecture.
    :param y: Actual values
    :return: L1 norm between predictions and y
    """
    if len(predictions) == 0:
        print("Predictions is empty list")
        return 0
    if len(predictions) != len(y):
        print("Predictions and y must have same size")
        return 0

    return sum(abs(predictions - y))/len(predictions)


def linfty_norm(predictions, y):
    """
    Computes the L_infinity norm between predictions made by the neural network and the actual values.
    :param predictions: Predictions made by the neural network architecture.
    :param y: Actual values
    :return: L_infinity norm between predictions and y
    """
    return max(abs(predictions - y))


def measure_performance(model: NeuralNetwork, test_data: CustomDataset, verbose=True):
    # Get features and labels
    test_features = test_data.features
    test_labels = test_data.labels

    # Get predictions
    with torch.no_grad():
        p_hat = np.array([])
        for input_values in test_features:
            prediction = model(input_values)
            p_hat = np.append(p_hat, prediction[0].item())

    # Get labels as np arrays
    p = np.array([])
    for value in test_labels:
        p = np.append(p, value[0].item())

    # Compute the norms
    delta_p_l1     = l1_norm(p_hat, p)
    delta_p_linfty = linfty_norm(p_hat, p)

    if verbose:
        print("Errors for p: %e  with L1 and %e with Linfty" % (delta_p_l1, delta_p_linfty))

    return delta_p_l1, delta_p_linfty


###########
# PRUNING #
###########


def delete_column(x: torch.tensor, index: int) -> torch.Tensor:
    """
    Prunes a matrix by deleting a column of the matrix.
    :param x: Torch tensor with shape (n, m)
    :param index: Index of column to be deleted.
    :return: Torch tensor with shape (n, m-1).
    """

    # Delete the column by splitting into two pieces, transpose the tensors for cat
    a = torch.transpose(x[:, :index], 0, 1)
    b = torch.transpose(x[:, index + 1:], 0, 1)

    # Concatenate the two results, with the desired column deleted
    new = torch.cat((a, b))

    return torch.transpose(new, 0, 1)


def delete_row_tensor(x, index):
    """
    Prunes a matrix by deleting a column of the matrix.
    :param x: Torch tensor with shape (n, m)
    :param index: Index of column to be deleted.
    :return: Torch tensor with shape (n-1, m).
    """

    # Delete the column by splitting into two pieces, transpose the tensors for cat
    a = x[:index]
    b = x[index + 1:]

    # Return concatenation
    return torch.cat((a, b))

def prune(old_model):
    """
    Prunes a neural network.
    :param old_model: Neural network which we want to prune
    :return: model: New neural network, pruned version of old model.
    """

    # Get the parameter values of the old model
    state_dict = old_model.state_dict()
    #state_dict_items = state_dict.items()

