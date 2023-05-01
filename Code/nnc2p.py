"""Python script for code related to neural networks used in the C2P conversion."""

from typing import Callable
import numpy as np
import pandas as pd
import os
import csv
import torch.optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
import data
import physics
from data import CustomDataset
from datetime import datetime
# Get appropriate directories
cwd = os.getcwd()  # "Code" folder
master_dir = os.path.abspath(os.path.join(cwd, ".."))  # master directory of this repo

if torch.cuda.is_available(): 
    DEVICE = "cuda:0"
    torch.set_default_device('cuda')
else: 
    DEVICE = "cpu"

##########################
# LOAD STANDARD DATASETS #
##########################

# Specify the desired CSV file locations of the "standard" train and test data already saved
TRAIN_SIZE = 80000
TEST_SIZE = 10000
# Standard training and testing datasets: 
TRAINING_DATA_CSV = data.read_training_data(os.path.join(master_dir, "Data/ideal_gas_c2p_train_data.csv"))
TEST_DATA_CSV     = data.read_training_data(os.path.join(master_dir, "Data/ideal_gas_c2p_test_data.csv"))
# Load them as CustomDatasets
TRAINING_DATA = data.CustomDataset(TRAINING_DATA_CSV)
TEST_DATA     = data.CustomDataset(TEST_DATA_CSV)
# Put this data into a DataLoader
TRAIN_DATALOADER = DataLoader(TRAINING_DATA, batch_size=32)
TEST_DATALOADER  = DataLoader(TEST_DATA, batch_size=32)

#####################
# AUXILIARY METHODS #
#####################


def generate_time_identifier() -> str:
    """
    Generates a unique identifier on a specific timestamp, up to the minute
    :return: String displaying time in format YYYY-MM-DD-HH-MM
    """

    # Get a datetime
    dt = datetime.now()

    # Get the string: format is: year_month_day_h_min_sec
    y      = str(dt.year)
    m      = str(dt.month)
    d      = str(dt.day)
    h      = str(dt.hour)
    minute = str(dt.minute)

    return y + "_" + m + "_" + d + "_" + h + "_" + minute


def write_to_txt(filename, text, verbose=True):
    """
    Small auxiliary file that writes a line to a txt file, used for logging progress in training or pruning.
    :param filename: Txt file to which we will write txt.
    :param text: Text that has to be written to the file.
    :param verbose: Boolean indicating whether we also should print the text to the screen.
    :return: Nothing.
    """
    if verbose:
        print(text)
    f = open(filename, "a")
    f.write(text + "\n")
    f.close()


def write_to_csv(csv_file, row):
    """
    Small auxiliary file that writes a line to a csv file, used for logging progress in training or pruning.
    :param csv_file: csv file to which we will write csv.
    :param row: Data that has to be written to the file.
    :return: Nothing.
    """
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        # Write data
        writer.writerow(row)


#########################
# NETWORK ARCHITECTURES #
#########################

### TODO replace by the version defined in the notebooks!
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


def get_hidden_sizes_from_state_dict(state_dict):
    """
    Finds the sizes of the two hidden layers of our 2-layer architecture given a state dict.
    :param state_dict: State dict of saved parameters
    :return: h1, size of first hidden layer, and h2, size of second hidden layer
    """
    h1 = np.shape(state_dict['linear1.bias'])[0]
    h2 = np.shape(state_dict['linear2.bias'])[0]

    return h1, h2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_nn(state_dict):
    """
    Create a NeuralNetwork object if given a dictionary of the weights, with correct sizes for hidden layers.
    :param state_dict: State dictionary containing the weights of the neural network
    :return:
    """
    h1, h2 = get_hidden_sizes_from_state_dict(state_dict)
    model = NeuralNetwork(h1=h1, h2=h2)
    model.load_state_dict(state_dict)

    return model


# The most recent and up to date, most flexible neural network we created:

class Net(nn.Module):
    """
    Implements a simple feedforward neural network.
    """

    def __init__(self, nb_of_inputs: int = 3, nb_of_outputs: int = 1, h: list = [600, 200], reg: bool = False,
                 activation_function=torch.nn.Sigmoid, output_bias=True) -> None:
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
                setattr(self, f"linear{i + 1}",
                        nn.Linear(self.h_augmented[i], self.h_augmented[i + 1], bias=output_bias))
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

############################
# GENERAL TRAINING ASPECTS #
############################

# def torch_mse(y: np.array, y_hat: np.array):  #  -> torch.Tensor
#     """
#     Compute the MSE, the mean squared error, between two numpy arrays.
#     """

#     return 

# TODO fix this!!!
def c2p_loss(prediction, original_cons):
    """
    Loss function which considers the mean squared error between the original conserved variables and the values
    for the conserved variables obtained by performing the P2C transformation on the prims obtained after the C2P conversion by the neural network.
    :param: prediction: Parameter returned by the neural network, i.e. the pressure.
    :param: original_cons: D, S, tau which was the input into the neural network. 
    """

    # Get the input values as separate variables
    D_value, S_value, tau_value = original_cons[:, 0], original_cons[:, 1], original_cons[:, 2]
    # Get all primitives
    v, W_value, eps_value, rho_value = physics.get_prims(D_value, S_value, tau_value, prediction)
    # Get predicted values:
    D_pred, S_pred, tau_pred = physics.p2c(rho_value, v, eps_value, prediction)
    print(D_pred)
    print(S_pred)
    print(tau_pred)
    # Save into new torch
    new_prims = torch.zeros_like(original_cons)
    new_prims[:, 0] = D_pred
    new_prims[:, 1] = S_pred
    new_prims[:, 2] = tau_pred

    loss_list = torch.mean((original_cons - new_prims)**2)

    return torch.mean(loss_list)

     
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
               report_progress: bool = False, use_c2p_loss: bool = False) -> None:
    """
    Does one epoch of the training loop.

    :param dataloader: Torch DataLoader object, containing training data.
    :param model: The neural network.
    :param loss_fn: The loss function on which the neural network is trained.
    :param optimizer: The optimization algorithm used.
    :param report_progress: Boolean indicating whether or not we want to print the progress of training.
    :param use_c2p_loss: Boolean indicating whether we compute loss through conservative variables.
    :return: None
    """
    # TODO - what's the issue here?
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction 
        prediction = model(X)
        # In case we use C2P loss function, have to provide conserved variables for the loss computation
        if use_c2p_loss:
            loss = compute_loss(prediction, X, loss_fn, model)
        else:
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

    def __init__(self, model: nn.Module, learning_rate: float, train_dataloader: DataLoader = None,
                 test_dataloader: DataLoader = None, loss_fn: Callable = nn.MSELoss(), optimizer = torch.optim.Adam,
                 train_losses=None, test_losses: list = None, adaptation_indices: list = None, use_c2p_loss: bool = False) -> None:
        """
        Initializes a Trainer object. This object brings together the neural network architecture defined above,
        the optimizer used, the dataloaders used while training, and simplifies the training process for the
        notebooks.

        :param model: The neural network.
        :param train_losses: List that contains the train losses during training, for plotting purposes.
        :param test_losses: List that contains the test losses during training, for plotting purposes.
        :param adaptation_indices: Indices at which the learning rate gets adapted, for plotting purposes.
        :param use_c2p_loss: Boolean indicating whether we compute loss through conservative variables.
        """

        # If None as argument given for train dataloader and test dataloader, make new sample
        if train_dataloader is None:
            train_dataloader = data.generate_dataloader(TRAIN_SIZE)

        if test_dataloader is None:
            test_dataloader = data.generate_dataloader(TEST_SIZE)

        # Save the model and optimizer as fields
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        # From the train dataloader, also save the normalization function
        # self.model.mean = train_dataloader.dataset.mean
        # self.model.std = train_dataloader.dataset.std
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.use_c2p_loss = use_c2p_loss
        self.optimizer = optimizer(model.parameters(), lr=learning_rate)
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

    def train(self, adaptation_threshold=0.9995, adaptation_multiplier=0.5, number_of_epochs: int = 500,
              log_file: str = "train_log.txt", csv_file: str = "train_log.csv"):

        # TODO - write doc

        # Initialize txt log file
        write_to_txt(log_file, f"Training the model for {number_of_epochs} epochs.")

        # Initialize csv file
        header = ['epoch', 'train_loss', 'test_loss', 'learning_rate']
        # Open the filename and write the data to it
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(header)

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
            write_to_txt(log_file, f"\n Epoch {epoch_counter} \n --------------")
            # Train the network
            train_loop(self.train_dataloader, self.model, self.loss_fn, self.optimizer, use_c2p_loss=self.use_c2p_loss)
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
                    self.learning_rate = adaptation_multiplier * self.learning_rate
                    write_to_txt(log_file, f"Adapting learning rate to {self.learning_rate}")
                    # Change optimizer
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
                    self.optimizer = optimizer
                    # Add the epoch time for plotting later on
                    self.adaptation_indices.append(epoch_counter)

            # Report progress:
            write_to_txt(log_file, "Train loss: %0.2E" % average_train_loss)
            write_to_txt(log_file, "Test  loss: %0.2E" % average_test_loss)
            # Save progress in CSV
            write_to_csv(csv_file, [epoch_counter, average_train_loss, average_test_loss, self.learning_rate])

            # Another epoch passed - increment counter
            counter += 1
            epoch_counter += 1

        print("Done!")
        
    def report_training(self, csv_filename, comment = " "):
        """Writes the performance (training) of a network to a CSV file such that we can improve architecture design. Saves the hidden layers, the activation functions
        The CSV file saves:
        1. Hidden layer details
        2. Activation functions
        3. Number of epochs trained
        4. Final training loss
        5. Final learning rate

        Args:
            csv_filename (str): Name of the CSV file to which we want to save the information to.

        Returns:
            None: Void return
        """
        
        activation_func = self.model.activation1.__class__.__name__
        final_lr = self.optimizer.param_groups[0]['lr']
        dat = [self.model.h, str(activation_func), len(self.train_losses), self.train_losses[-1], final_lr, comment]
        write_to_csv(csv_filename, dat)

###########################
# PERFORMANCE OF NETWORKS #
###########################


# TODO - type hinting?
def l1_norm(predictions, y, reduction = True):
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

    if reduction:
        return np.sum(abs(predictions - y), axis=0)/len(predictions)
    else:
        return abs(predictions - y)


def linfty_norm(predictions, y):
    """
    Computes the L_infinity norm between predictions made by the neural network and the actual values.
    :param predictions: Predictions made by the neural network architecture.
    :param y: Actual values
    :return: L_infinity norm between predictions and y
    """
    return np.max(abs(predictions - y), axis=0)


def l2_norm(predictions, y, reduction = True):
    """
    Computes the L_infinity norm between predictions made by the neural network and the actual values.
    :param predictions: Predictions made by the neural network architecture.
    :param y: Actual values
    :return: L_infinity norm between predictions and y
    """
    if reduction:
        return np.mean(abs(predictions - y)**2, axis=0)
    else:
        return abs(predictions - y) ** 2


def measure_performance(model: nn.Module, test_data: Dataset = None, verbose=False):
    """
    Measures the performance similar to how the Dieslhorst et al. paper does it. Computes the L1-norm and Linfty-norm
    errors (difference between the predictions and true values) on the provided test set.

    :param model: Neural network.
    :param test_data: Test data in DataLoader object.
    :param verbose: Show the errors by printing them directly to the screen.
    :return: delta_p_l1, delta_p_l2, delta_p_linfty: The computed errors in respective norms.
    """
    if test_data is None:
        test_data = data.generate_dataloader(TEST_SIZE)

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
        print("Errors: %e  with L1, %e  with L2 and %e with Linfty" % (delta_p_l1, delta_p_l2, delta_p_linfty))

    return delta_p_l1, delta_p_l2, delta_p_linfty


def measure_performance_size(model: NeuralNetwork, size_test_data: int = 1000, verbose=False):
    """
    Measures the performance similar to how the Dieselhorst et al. paper does it. Computes the L1-norm and Linfty-norm
    errors (difference between the predictions and true values) on the provided test set.

    :param model: Neural network.
    :param size_test_data: Number of test data points on which we test the performance.
    :param verbose: Show the errors by printing them directly to the screen.
    :return: delta_p_l1, delta_p_l2, delta_p_linfty: The computed errors in respective norms.
    """

    # Generate test data of specified size using physics package

    test_data = data.CustomDataset(physics.generate_data_as_df(number_of_points=size_test_data))
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


def prune(old_model: nn.Module, layer_index: int, neuron_index: int) -> NeuralNetwork:
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
    #h1 = old_model.h1
    #h2 = old_model.h2

    # Updated code:
    h1 = old_model.h[0]
    h2 = old_model.h[1]

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

    # This depends on the implementation of the
    ac_func = old_model.activation_function
    out_bias = old_model.output_bias
    new_model = Net(h=[h1, h2], nb_of_outputs=3, activation_function=ac_func, output_bias=old_model.output_bias).double()

    new_model.activation_function = ac_func
    new_model.output_bias = out_bias

    # Now, load the pruned model's weights into the architecture
    new_model.load_state_dict(new_state_dict)

    return new_model


def find_optimal_neuron_to_prune(model: nn.Module, validation_data, verbose=True) -> tuple[int, int, float, dict]:
    """
    Finds the optimal neuron to prune, that is, the neuron which after pruning this neuron from the current model,
    gives the model which has the best performance over all pruned models.

    :param model: Current model which we wish to prune
    :param validation_data_size: Number of data points to be used in the validation set.
    :param verbose: Boolean indicating whether we wish to print the progress.
    :return: best_layer_index, best_neuron_index, best_performance, values_dict: The layer and neuron index of the neuron
     that gives best model after pruning and its performance after pruning. Values dict is useful for plotting this function.
    """
    # Initialize the return values
    best_layer_index  = 0
    best_neuron_index = 0

    # To keep track of best L2 norm, initialize with a high value
    best_performance = 9999

    # Initialize empty dicts to save values
    values_dict = {'layer_index': [], 'neuron_index': [], 'l1_norm': [], 'l2_norm': [], 'linfty_norm': []}

    # Get the validation set to compare performances, 1000 datapoints are tested
    #validation_data = data.CustomDataset(physics.generate_data_as_df(validation_data_size))


    # Start form an existing model
    current_model = model

    # Get the sizes of hidden layers of this model
    #h1 = current_model.h1
    #h2 = current_model.h2
    h1 = current_model.h[0]
    h2 = current_model.h[1]
    hidden_layer_sizes = [h1, h2]

    # Loop over all layers:
    for i, layer_index in enumerate([1, 2]):
        # Loop over all neurons of a layer
        if verbose:
            print('\nLayer index: ' + str(layer_index) + "\n")
        for neuron_index in range(hidden_layer_sizes[i]):
            # Show progress
            if verbose:
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
            values_dict['l1_norm'].append(l1)
            values_dict['l2_norm'].append(l2)
            values_dict['linfty_norm'].append(linfty)

    return best_layer_index, best_neuron_index, best_performance, values_dict


def hill_climbing_pruning(model, max_pruning_number, lr: float = 1e-6, validation_data_size: int = 1000, train_data_size: int = 40000,
                          test_data_size: int = 10000, l2_threshold: float = 4e-7, nb_of_train_epochs: int = 100):

    # Prepare everything for saving progress: make directory, create empty txt file for logging
    identifier = generate_time_identifier()
    print(f"Saving pruning progress with ID: {identifier}")

    # Make new directory to save files:
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, "Pruning_" + identifier)
    os.mkdir(save_dir)

    # Create empty log files
    txt_file = os.path.join(save_dir, "log.txt")
    csv_file = os.path.join(save_dir, "log.csv")

    # Already write header to the csv
    header = ['counter', 'layer_index', 'neuron_index', 'performance']
    # Open the filename and write the data to it
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(header)

    # Initialize a counter
    counter = 1

    while counter < max_pruning_number:
        # Next iteration of pruning to do
        write_to_txt(txt_file, f"======== Pruning iteration {counter}/{max_pruning_number} ========")
        # Find best neuron to prune
        best_layer_index, best_neuron_index, _, values_dict = find_optimal_neuron_to_prune(model, validation_data_size=validation_data_size)
        # Prune that neuron
        model = prune(model, best_layer_index, best_neuron_index)
        # Test the performance on a larger dataset to have a statistically more accurate measure of its performance before
        # we decide to retrain this model
        _, best_performance, _ = measure_performance_size(model, size_test_data=10000, verbose=True)
        # Show progress and save to the txt log file
        write_to_txt(txt_file, f"Pruned {counter}/{max_pruning_number}. Performance is {best_performance}")
        write_to_csv(csv_file, [counter, best_layer_index, best_neuron_index, best_performance])
        if best_performance > l2_threshold:
            # If the model is too bad after pruning, retrain it

            write_to_txt(txt_file, f"Performance dropped too much, retraining the model for {nb_of_train_epochs} epochs.")
            # Generate training data
            train_dataloader = data.generate_dataloader(train_data_size)
            test_dataloader = data.generate_dataloader(test_data_size)
            # Create trainer object and train it
            trainer = Trainer(model, lr, train_dataloader, test_dataloader)
            # Specify where we log the training procedure for this model
            train_txt = os.path.join(save_dir, "training_log" + str(counter) + ".txt")
            train_csv = os.path.join(save_dir, "training_log" + str(counter) + ".csv")
            trainer.train(number_of_epochs=nb_of_train_epochs, log_file=train_txt, csv_file=train_csv)
            # After retraining, get the performance again (the L2 norm, so second return argument)
            _, best_performance, _ = measure_performance_size(model)
            write_to_txt(txt_file, f"Retrained {counter}/{max_pruning_number}. Performance is {best_performance}")
            write_to_csv(csv_file, [counter, best_layer_index, best_neuron_index, best_performance])

        # After pruning and possibly training, save the new model's parameters:
        torch.save(model.state_dict(), os.path.join(save_dir, str(counter) + ".pth"))

        # Increment counter
        counter += 1

    # Save the final model separately as well
    torch.save(model.state_dict(), os.path.join(save_dir, "pruned.pth"))

    return model

###################
# POST-PROCESSING #
###################


def evaluate_models(models: list, test_data: Dataset, csv_file: str = "performance_models.csv") -> pd.DataFrame:
    """
    Evaluates a list of models and saves this evaluation to a specified CSV file and returns it as a Pandas DataFrame.
    It also specifies the compression ratio, assuming the first element to be the largest model to which we compare
    subsequent model sizes.
    :param models: List of neural networks which we wish to evaluate and compare to one another
    :param csv_file: The location of the CSV file where we wish to save the results
    :param size_test_data: Size of dataset used to evaluate the performance of the networks.
    :return: Pandas DataFrame showing the data, for ease of visualization later on in notebooks.
    """
    # Initialize CSV file
    header = ['h1', 'h2', 'N', 'l1', 'l2', 'linfty', 'nb_params', 'compression_ratio']
    # Open the filename and write the data to it
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(header)

    # Dataset to evaluate the models
    #test_data = data.CustomDataset(physics.generate_data_as_df(number_of_points=size_test_data))

    # Go over all models, evaluate performance
    for i, model in enumerate(models):
        h1 = model.h1
        h2 = model.h2
        N = h1 + h2
        l1, l2, linfty = measure_performance(model, test_data)
        nb_params = count_parameters(model)
        if i == 0:
            largest = nb_params
        compression_ratio = nb_params/largest
        write_to_csv(csv_file, [h1, h2, N, l1, l2, linfty, nb_params, compression_ratio])

    return pd.read_csv(csv_file)


#################
# EXPORT MODELS #
#################

def export_model(file_name, save_name, weights_names=["weight0", "bias0", "weight2", "bias2", "weight4", "bias4"]):
    """
    Exports a model located at filename. By default, just saves the "flat" versions. See export_models notebook
    for older versions where we also exported other versions.
    :param file_name: Location of the model, typically in the Models folder of the repo.
    :param save_name: Location where the CSV files should be saved. This should be a directory
    :return: None, but saves the parameter values as CSV to the desired location.
    """

    # Make directory if save_name not found
    if not(os.path.exists(save_name) and os.path.isdir(save_name)):
        print(f"Directory not found. Making new one at {save_name}")
        os.mkdir(save_name)

    # Load the model
    model = torch.load(file_name)
    # State dict contains all the variables
    state_dict = model.state_dict().items()
    # Names to save the files:
    flat_save_names = [os.path.normpath(save_name + f"/{name}_flat.csv") for name in weights_names]

    # Save each one:
    counter = 0
    for param_name, item in state_dict:
        # Get appropriate name
        flat_save_name = flat_save_names[counter]

        # Get the matrix and flatten it as well
        matrix_np = item.numpy()
        flat_matrix_np = matrix_np.flatten()

        # Save to txt
        np.savetxt(flat_save_name, flat_matrix_np, delimiter=",", newline=',\n', fmt="%0.35f")

        counter += 1

    print(f"Succesfully exported model parameters to CSV file, at {save_name}")