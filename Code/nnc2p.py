"""Python script for code related to neural networks used in the C2P conversion."""

from typing import Callable
import numpy as np
import pandas as pd
import torch.optim
from torch import nn
from torch.utils.data import DataLoader
import data
import physics
from data import CustomDataset

#################
# LOAD DATASETS #
#################

# Specify the desired CSV file locations of the "standard" train and test data already saved
TRAINING_DATA_CSV = data.read_training_data("D:/Coding/master-thesis-AI/data/NNC2P_data_train.csv")
TEST_DATA_CSV     = data.read_training_data("D:/Coding/master-thesis-AI/data/NNC2P_data_test.csv")
# Load them as CustomDatasets
TRAINING_DATA = data.CustomDataset(TRAINING_DATA_CSV)
TEST_DATA     = data.CustomDataset(TEST_DATA_CSV)
# Put this data into a DataLoader
TRAIN_DATALOADER = DataLoader(TRAINING_DATA, batch_size=32)
TEST_DATALOADER  = DataLoader(TEST_DATA, batch_size=32)


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

        # For convenience, save the sizes of the hidden layers as fields as well
        self.h1 = h1
        self.h2 = h2

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
        stack_0_bias = state_dict['stack.0.bias']
        self.linear1.weight = nn.parameter.Parameter(stack_0_weight)
        self.linear1.bias = nn.parameter.Parameter(stack_0_bias)

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

    def __init__(self, train_dataloader: DataLoader, test_dataloader: DataLoader, model: NeuralNetwork, learning_rate: float,
                 loss_fn: Callable = nn.MSELoss(), train_losses=None, test_losses: list = None,
                 adaptation_indices: list = None) -> None:
        """
        Initializes a Trainer object. This object brings together the neural network architecture defined above,
        the optimizer used, the dataloaders used while training, and simplifies the training process for the
        notebooks.

        :param model: The neural network.
        :param train_losses: List that contains the train losses during training, for plotting purposes.
        :param test_losses: List that contains the test losses during training, for plotting purposes.
        :param adaptation_indices: Indices at which the learning rate gets adapted, for plotting purposes.
        """
        # Save the model and optimizer as fields
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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


def l2_norm(predictions, y):
    """
    Computes the L_infinity norm between predictions made by the neural network and the actual values.
    :param predictions: Predictions made by the neural network architecture.
    :param y: Actual values
    :return: L_infinity norm between predictions and y
    """
    return sum(abs(predictions - y)**2)/len(predictions)


def measure_performance(model: NeuralNetwork, test_data: CustomDataset, verbose=False):
    """
    Measures the performance similar to how the Dieslhorst et al. paper does it. Computes the L1-norm and Linfty-norm
    errors (difference between the predictions and true values) on the provided test set.

    :param model: Neural network.
    :param test_data: Test data in DataLoader object.
    :param verbose: Show the errors by printing them directly to the screen.
    :return: delta_p_l1, delta_p_l2, delta_p_linfty: The computed errors in respective norms.
    """
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
    delta_p_l2     = l2_norm(p_hat, p)
    delta_p_linfty = linfty_norm(p_hat, p)

    if verbose:
        print("Errors: %e  with L1, %e  with L2 and %e with Linfty" % (delta_p_l1, delta_p_l1, delta_p_linfty))

    return delta_p_l1, delta_p_l2, delta_p_linfty


def measure_performance_size(model: NeuralNetwork, size_test_data: int = 1000, verbose=False):
    """
    Measures the performance similar to how the Dieslhorst et al. paper does it. Computes the L1-norm and Linfty-norm
    errors (difference between the predictions and true values) on the provided test set.

    :param model: Neural network.
    :param size_test_data: Number of test data points on which we test the performance.
    :param verbose: Show the errors by printing them directly to the screen.
    :return: delta_p_l1, delta_p_l2, delta_p_linfty: The computed errors in respective norms.
    """

    # Generate test data of specified size using physics package

    test_data = data.CustomDataset(physics.generate_data_as_df(size_test_data))
    # Measure performance on this test data
    return measure_performance(model, test_data, verbose=verbose)

###########
# PRUNING #
###########


def delete_column(x: torch.tensor, index: int) -> torch.tensor:
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


def delete_row(x: torch.tensor, index: int) -> torch.tensor:
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


def prune_bias(b: torch.tensor, index: int) -> torch.tensor:
    """
    Prunes the bias vector, by deleting the entry of the bias vector at specified index.
    :param b: The original bias vector
    :param index: The index position of the element to be removed from the pruned vector.
    :return: torch.tensor: New bias vector with specified element removed.
    """

    # Get everything ocurring before and after the index position
    before = b[:index]
    after = b[index + 1:]

    # Return the concatenation
    return torch.cat([before, after])


def prune(old_model: NeuralNetwork, layer_index: int, neuron_index: int) -> NeuralNetwork:
    """
    Prunes a neural network once given the hidden layer index and the neuron index. Note: we assume two layers.

    :param old_model: Old neural network which is going to be pruned soon.
    :param layer_index: Index of the hidden layer where the neuron to be pruned is located.
    :param neuron_index: Index position of the neuron to be pruned within its hidden layer.
    :return: NeuralNetwork: pruned version
    """

    # Get the weights and biases of the old model
    state_dict = old_model.state_dict()
    # Copy the state dict of original model
    new_state_dict = state_dict.copy()

    # For convenience, get the sizes of the hidden layers from the old network
    h1 = old_model.h1
    h2 = old_model.h2

    # Based on the layer_index, get the correct layers' parameters' names:
    this_layer_name = "linear" + str(layer_index)
    next_layer_name = "linear" + str(layer_index + 1)

    this_weight_name = this_layer_name + ".weight"
    this_bias_name = this_layer_name + ".bias"
    next_weight_name = next_layer_name + ".weight"

    # Now, get these parameters:
    this_weight = state_dict[this_weight_name]
    this_bias = state_dict[this_bias_name]
    next_weight = state_dict[next_weight_name]

    # Prune the matrices:
    this_weight = delete_row(this_weight, neuron_index)
    this_bias = prune_bias(this_bias, neuron_index)
    next_weight = delete_column(next_weight, neuron_index)

    # Update these parameters in the new state dict
    new_state_dict[this_weight_name] = this_weight
    new_state_dict[this_bias_name] = this_bias
    new_state_dict[next_weight_name] = next_weight

    # Get the new model. First, instantiate a random architecture, with correct sizes
    if layer_index == 1:
        h1 = h1 - 1
    else:
        h2 = h2 - 1

    new_model = NeuralNetwork(h1=h1, h2=h2)

    # Now, load the pruned model's weights into the architecture
    new_model.load_state_dict(new_state_dict)

    return new_model


def find_optimal_neuron_to_prune(model: NeuralNetwork, validation_data_size: int = 1000) -> tuple[int, int, float, dict]:
    """
    Finds the optimal neuron to prune, that is, the neuron which after pruning this neuron from the current model,
    gives the model which has the best performance over all pruned models.

    :param model: Current model which we wish to prune
    :param validation_data_size: Number of data points to be used in the validation set.
    :return: best_layer_index, best_neuron_index: The layer and neuron index of the neuron that gives best model
    after pruning.
    """
    # Initialize the return values
    best_layer_index  = 0
    best_neuron_index = 0

    # To keep track of best L2 norm, initialize with a high value
    best_performance = 9999

    # Initialize empty dicts to save values
    values_dict = {'layer_index': [], 'neuron_index': [], 'l1_norm': [], 'l2_norm': [], 'linfty_norm': []}

    # Get the validation set to compare performances, 1000 datapoints are tested
    validation_data = data.create_customdataset(physics.generate_data_as_df(validation_data_size))

    # Start form an existing model
    current_model = model

    # Get the sizes of hidden layers of this model
    h1 = current_model.h1
    h2 = current_model.h2
    hidden_layer_sizes = [h1, h2]

    # Loop over all layers:
    for i, layer_index in enumerate([1, 2]):
        # Loop over all neurons of a layer
        print('\nLayer index: ' + str(layer_index) + "\n")
        for neuron_index in range(hidden_layer_sizes[i]):
            # Show progress
            print(f"Neuron index: {neuron_index} out of {hidden_layer_sizes[i]}", end="\r")
            # Prune the model with that layer index and neuron index
            new_model = prune(current_model, layer_index, neuron_index)
            # Get the performance of that model
            l1, l2, linfty = measure_performance(new_model, validation_data)
            # TODO - compare norm specified as argument or will we always use L2 norm?
            # Compare the performance of the model with the current best value
            if l2 < best_performance:
                # If better, save as new best value
                best_layer_index  = layer_index
                best_neuron_index = neuron_index
                best_performance = l2

            # Save results
            values_dict['layer_index'].append(layer_index)
            values_dict['neuron_index'].append(neuron_index)
            values_dict['l1_norm'].append(l1_norm)
            values_dict['l2_norm'].append(l2_norm)
            values_dict['linfty_norm'].append(linfty_norm)

    return best_layer_index, best_neuron_index, best_performance, values_dict


def hill_climbing_pruning(model, max_pruning_number, validation_data_size: int = 1000, l2_threshold: float = 10e-6, nb_of_train_epochs: int = 100):

    counter = 0
    while counter < max_pruning_number:
        # Find best neuron to prune
        best_layer_index, best_neuron_index, best_performance, values_dict = find_optimal_neuron_to_prune(model, validation_data_size=validation_data_size)
        # Prune that neuron
        model = prune(model, best_layer_index, best_neuron_index)
        counter += 1
        print(f"Pruned {counter}/{max_pruning_number}. Performance is {best_performance}")
        # If the model is too bad after pruning, retrain it
        if best_performance > l2_threshold:
            print(f"Performance drop. Retraining.")
            # Create new Trainer object
            trainer = Trainer(TRAIN_DATALOADER, TEST_DATALOADER, model, 1e-4)
            trainer.train(number_of_epochs=nb_of_train_epochs)

    return model





