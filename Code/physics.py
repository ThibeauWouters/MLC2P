"""
This script is dedicated to generating the data samples used for training and testing the neural networks.
"""
import random
import csv
import data
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
# from data import CustomDataset
import os
import h5py

if torch.cuda.is_available():
    DEVICE = "cuda:0"
    torch.set_default_device('cuda')
else:
    DEVICE = "cpu"

###################
# HYPERPARAMETERS #
###################

# Define ranges of parameters for training the ideal gas law (see paper Section 2.1)
RHO_MIN = 0
RHO_MAX = 10.1
EPS_MIN = 0
EPS_MAX = 2.02
V_MIN = 0
V_MAX = 0.721
# TODO - these ranges are valid for the Alfven wave!
B_MIN = -2.1
B_MAX = 2.1
# Get master folder
cwd = os.getcwd()  # this refers to the "Code" folder
master_dir = os.path.abspath(os.path.join(cwd, ".."))  # this refers to the root folder


###########################
# BASIC PHYSICS FUNCTIONS #
###########################


# Methods to compute physical quantities. See Dieselhorst et al. paper for more information.
def ideal_eos(rho: float, eps: float, gamma: float = 5 / 3) -> float:
    """
    See eq2 Dieselhorst et al. paper. Computes the EOS being an analytic Gamma law
    """
    return (gamma - 1) * rho * eps


def h(rho: float, eps: float, v: float, p: float) -> float:
    """
    See eq2 Dieselhorst et al. paper. Computes enthalpy
    """
    return 1 + eps + p / rho


def W(rho: float, eps: float, v, p: float = None) -> float:
    """
    See eq2 Dieselhorst et al. paper. Lorentz factor. Here, in 1D so v = v_x
    """
    if isinstance(v, float):
        v_sqr = v ** 2
    else:
        # v must be numpy array for 2D or 3D
        v_sqr = np.sum(v ** 2)

    return (1 - v_sqr) ** (-1 / 2)



def D(rho: float, eps: float, v: float, p: float) -> float:
    """
    See eq2 Dieselhorst et al. paper.
    """
    return rho * W(rho, eps, v, p)


def S(rho: float, eps: float, v: float, p: float) -> float:
    """
    See eq2 Dieselhorst et al. paper.
    """
    return rho * h(rho, eps, v, p) * ((W(rho, eps, v, p)) ** 2) * v


def tau(rho: float, eps: float, v: float, p: float) -> float:
    """
    See eq2 Dieselhorst et al. paper.
    """
    return rho * (h(rho, eps, v, p)) * ((W(rho, eps, v, p)) ** 2) - p - D(rho, eps, v, p)


def p2c(rho: float, eps: float, v, p: float):
    """
    Performs the P2C transformation, as given by the Eq. 2 of Dieselhorst et al. paper.
    """
    d = D(rho, eps, v, p)
    s = S(rho, eps, v, p)
    t = tau(rho, eps, v, p)

    return d, s, t


def p2c_GRMHD(rho: float, eps: float, vx: float, vy: float, vz: float, Bx: float, By: float, Bz: float):

    # Get auxiliary variables first
    B_sqr = Bx ** 2 + By ** 2 + Bz ** 2  # square of B vector
    v_sqr = vx ** 2 + vy ** 2 + vz ** 2  # square of v vector
    B_dot_v = Bx * vx + By * vy + Bz * vz  # dot product of B and v
    lfac = (1 - v_sqr) ** (-1 / 2)  # Lorentz factor
    alpha_b0 = lfac * B_dot_v  # lapse function times zero component small b (Gmunu paper eq 14)
    b_sqr = B_sqr/(lfac**2) + B_dot_v ** 2  # small b squared (Gmunu paper, eq 15)
    bx = Bx / lfac + alpha_b0 * vx
    by = By / lfac + alpha_b0 * vy  # small b see underneath eq 15, index is lowercase!
    bz = Bz / lfac + alpha_b0 * vz

    # Get pressure and magnetically modified enthalpy (latter: see under eq 7)
    p = ideal_eos(rho, eps)
    p_star = p + b_sqr/2
    h_star = 1 + eps + (p + b_sqr)/rho  # magnetically modified enthalpy

    # Get the cons now
    d   = rho * lfac
    sx  = rho * h_star * (lfac ** 2) * vx - alpha_b0 * bx
    sy  = rho * h_star * (lfac ** 2) * vy - alpha_b0 * by
    sz  = rho * h_star * (lfac ** 2) * vz - alpha_b0 * bz
    tau = rho * h_star * (lfac ** 2) - p_star - (alpha_b0 ** 2) - d

    return d, sx, sy, sz, tau


# def cons_add_EM(d, sx, sy, sz, tau, vx, vy, vz, Bx, By, Bz):
#     """Adds the magnetic contributions to the conserved variables. See Kastaun paper, eqs (18) -- (21).
#         Note that we assume 3D."""
#
#     # Define auxiliary terms
#     B_sqr = Bx ** 2 + By ** 2 + Bz ** 2
#     v_sqr = vx ** 2 + vy ** 2 + vz ** 2
#     B_dot_v = vx * Bx + vy * By + vz * Bz
#
#     # Get E terms
#     E_sqr = B_sqr * v_sqr - (B_dot_v) ** 2
#
#     # To add:
#     delta_tau = 0.5 * (E_sqr + B_sqr)
#
#     delta_Sx = B_sqr * vx - B_dot_v * Bx
#     delta_Sy = B_sqr * vy - B_dot_v * By
#     delta_Sz = B_sqr * vz - B_dot_v * Bz
#
#     # Now add them
#     sx += delta_Sx
#     sy += delta_Sy
#     sz += delta_Sz
#
#     tau += delta_tau
#
#     return d, sx, sy, sz, tau


def get_prims(D_value: float, S_value: float, tau_value: float, p: float) -> tuple[float, float, float, float]:
    v = S_value / (tau_value + D_value + p)
    W_value = 1 / np.sqrt(1 - v ** 2)
    eps_value = (tau_value + D_value * (1 - W_value) + p * (1 - W_value ** 2)) / (D_value * W_value)
    rho_value = D_value / W_value

    return v, W_value, eps_value, rho_value


def c2c(D_value: float, S_value: float, tau_value: float, model: nn.Module, nb_repetitions: int = 1):
    # Make a copy for clarity -- we are going to override these variables later on
    d = D_value
    s = S_value
    t = tau_value

    for _ in range(nb_repetitions):
        # Repeat the chain several times
        with torch.no_grad():
            # Get the pressure using the neural network
            press = model(torch.tensor([d, s, t]).float())
            # press is a tensor, convert to a float
            press = press.item()

        # From cons and pressure, compute other prims
        v, _, eps, rho = get_prims(d, s, t, press)
        d, s, t = p2c(rho, eps, v, press)

    return d, s, t


def c2c_dataset(dataset: Dataset, model: nn.Module, nb_repetitions: int = 1):
    # Measure the performance/robustness of a model on a whole dataset, return error

    l1_error = []
    l2_error = []
    # TODO linfty error norm?

    # Iterate over all test examples
    cons_dataset = dataset.features
    for i in range(len(cons_dataset)):
        D_value, S_value, tau_value = cons_dataset[i][0].item(), cons_dataset[i][1].item(), cons_dataset[i][2].item()
        # Get the version after certain nb of repetitions
        D_prime, S_prime, tau_prime = c2c(D_value, S_value, tau_value, model, nb_repetitions=nb_repetitions)
        # Compute errors
        cons = np.array([D_value, S_value, tau_value])
        cons_prime = np.array([D_prime, S_prime, tau_prime])
        l1_error.append(abs(cons - cons_prime))
        l2_error.append((cons - cons_prime) ** 2)

    return l1_error, l2_error


def p2p(rho: float, eps: float, v: float, model: nn.Module, nb_repetitions: int = 1):
    press = ideal_eos(rho, eps)

    for _ in range(nb_repetitions):
        # Compute the cons from the prims
        d, s, t = p2c(rho, eps, v, press)
        with torch.no_grad():
            # Use cons and neural net to get pressure
            press = model(torch.tensor([d, s, t]).float())
        # Get new primitives
        v, _, eps, rho = get_prims(d, s, t, press)

    return press


#############################
# DATA GENERATING FUNCTIONS #
#############################


def generate_c2p_data_ideal_gas(number_of_points: int = 10000, save_name: str = "",
                                rho_min=RHO_MIN, rho_max=RHO_MAX,eps_min=EPS_MIN, eps_max=EPS_MAX,
                                v_min=V_MIN, v_max=V_MAX,):
    """
    Generates training data of specified size, with ideal gas EOS, by sampling and performing the P2C transformation.
    :param number_of_points: The number of data points to be generated.
    :param save_name: In case we save the data, the name of the .csv file to which the data is saved.
    :return: list of which rows of sampled data of conserved and primitive variables.
    """

    # Initialize empty data
    data = []

    # Sample and generate a new row for the training data
    for i in range(number_of_points):
        # Generate primitive variables
        rho = random.uniform(rho_min, rho_max)
        eps = random.uniform(eps_min, eps_max)
        v = random.uniform(v_min, v_max)

        # Use above transformations to compute the pressure and conserved variables D, S, tau
        # Compute the pressure using an ideal gas law
        p = ideal_eos(rho, eps)
        # Do the P2C transformation
        Dvalue, Svalue, tauvalue = p2c(rho, eps, v, p)

        # Add the values to a new row
        new_row = [rho, eps, v, p, Dvalue, Svalue, tauvalue]

        # Append the row to the list
        data.append(new_row)

    header = ['rho', 'eps', 'v', 'p', 'D', 'S', 'tau']
    # Done generating data, now save if wanted by the user:
    if len(save_name) > 0:
        # Save as CSV, specify the header

        # Get the correct filename, pointing to the "data" directory
        filename = 'D:/Coding/master-thesis-AI/data' + save_name
        if (save_name[-3:]) != ".csv":
            # Make sure the file extension is csv
            filename += ".csv"

        # Open the filename and write the data to it
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            # write header
            writer.writerow(header)
            # write data
            writer.writerows(data)

    # Also return the data to the user
    return data, header


def generate_c2p_data_ideal_gas_GRMHD(number_of_points: int = 10000, save_name: str = "",
                                rho_min=RHO_MIN, rho_max=RHO_MAX,eps_min=EPS_MIN, eps_max=EPS_MAX,
                                v_min=V_MIN, v_max=0.5, b_min=-2, b_max=2) -> list:
    """
    Generates training data of specified size, with ideal gas EOS, by sampling and performing the P2C transformation.

    This is for 3D and including magnetic effects. See Kastaun paper for the specific equations that modify GRHD.
    :param number_of_points: The number of data points to be generated.
    :param save_name: In case we save the data, the name of the .csv file to which the data is saved.
    :return: list of which rows of sampled data of conserved and primitive variables.
    """

    # Initialize empty data
    data = []

    # Sample and generate a new row for the training data
    for i in range(number_of_points):
        # Generate primitive variables
        rho = random.uniform(rho_min, rho_max)
        eps = random.uniform(eps_min, eps_max)
        vx = random.uniform(v_min, v_max)
        vy = random.uniform(v_min, v_max)
        vz = random.uniform(v_min, v_max)
        Bx = random.uniform(b_min, b_max)
        By = random.uniform(b_min, b_max)
        Bz = random.uniform(b_min, b_max)

        # Use above transformations to compute the pressure and conserved variables D, S, tau
        p = ideal_eos(rho, eps)
        d, sx, sy, sz, tau = p2c_GRMHD(rho, eps, vx, vy, vz, Bx, By, Bz)
        # Add the values to a new row
        new_row = [rho, eps, vx, vy, vz, p, Bx, By, Bz, d, sx, sy, sz, tau]

        # Append the row to the list
        data.append(new_row)

    header = ['rho', 'eps', 'vx', 'vy', 'vz', 'p', 'Bx', 'By', 'Bz', 'D', 'Sx', 'Sy', 'Sz', 'tau']
    # Done generating data, now save if wanted by the user:
    if len(save_name) > 0:
        # Save as CSV, specify the header

        # Get the correct filename, pointing to the "data" directory
        filename = 'D:/Coding/master-thesis-AI/data' + save_name
        if (save_name[-3:]) != ".csv":
            # Make sure the file extension is csv
            filename += ".csv"

        # Open the filename and write the data to it
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            # write header
            writer.writerow(header)
            # write data
            writer.writerows(data)

    # Also return the data to the user
    return data,header


#################
# TABULATED EOS #
#################


def read_eos_table(filename: str):
    # Convert to proper filepath towards eos tables
    filename = os.path.join("eos_tables", filename)
    # Append it to the master dir
    filename = os.path.join(master_dir, filename)

    return h5py.File(filename, 'r')

# TODO - provide documentation and explain how this is different from the function below
def convert_eos_table(eos_table, var_names=["logenergy", "logpress", "cs2"], save_name="train_eos_table.h5"):
    """
    Convert the EOS table to rows, but each variable is one column
    That is, the format of the EOS tables as provided on the website treat all variables as different datasets in the .h5 files. Here, we convert these to a different format:
    """

    # Get the "index" or "input" variables (rho, temp, ye).
    rho, temp, ye = eos_table["logrho"][()], eos_table["logtemp"][()], eos_table["ye"][()]

    # Save the values to a dict for lookup
    values_dict = {}
    for name in var_names:
        values_dict[name] = eos_table[name][()]

    # Get lengths
    n_rho = len(rho)
    n_temp = len(temp)
    n_ye = len(ye)

    size = n_rho * n_temp * n_ye

    # We are going to save the values as columns later on
    rho_array = np.empty(size)
    temp_array = np.empty(size)
    ye_array = np.empty(size)

    # Get a dict to save the columns of the "output" variables
    var_dict = {}
    for name in var_names:
        var_dict[name] = np.empty(size)

    counter = 0

    # Loop and get values
    for i, r in enumerate(rho):
        for j, t in enumerate(temp):
            for k, y in enumerate(ye):
                # Start by saving the "input" values of this example
                rho_array[counter]  = r
                temp_array[counter] = t
                ye_array[counter]   = y
                # Add the "output" columns
                for name in var_names:
                    var_dict[name][counter] = values_dict[name][k, j, i]
                # Added an example, increment counter
                counter += 1

    # Save examples as HDF5 file
    with h5py.File(save_name, 'w') as f:
        # Save the examples under "my dataset"
        dataset = f.create_dataset('var_names', data=np.array(var_names))
        dataset = f.create_dataset('logrho', data=rho_array)
        dataset = f.create_dataset('logtemp', data=temp_array)
        dataset = f.create_dataset('ye', data=ye_array)
        for name in var_names:
            dataset = f.create_dataset(name, data=var_dict[name])

def generate_eos_data(eos_table, var_names=["logenergy", "logpress", "cs2"], save_name="train_eos_table.h5"):
    """
    Convert the EOS table to rows of training examples, taking the provided variables into account for the output.
    That is, the format of the EOS tables as provided on the website treat all variables as different datasets in the .h5 files. Here, we convert these to a different format:
    the result of this function is a saved .h5 file which contains rows of examples: the first three values correspond to (logrho, logtemp, ye) and are the "input" data, while the
    remaining values correspond to the desired "output" variables of a neural network trying to do regression in this EOS table and for the provided variable names. This can then easily
    be fed into a custom PyTorch dataset such that the whole table can get processed during training.
    Note: the order is reversed compared to the original EOS tables. That is, the EOS tables use (ye, T, rho), but here we use (rho, T, ye).
    """

    # Get the "index" or "input" variables (rho, temp, ye).
    rho, temp, ye = eos_table["logrho"][()], eos_table["logtemp"][()], eos_table["ye"][()]

    # Get a dict to save the columns of the "output" variables
    var_dict = {}
    for name in var_names:
        var_dict[name] = eos_table[name][()]
    # Fill an array of values
    features_array = []
    labels_array = []
    for i, r in enumerate(rho):
        for j, t in enumerate(temp):
            for k, y in enumerate(ye):
                # Start by saving the "input" values of this example
                features_array.append([r, t, y])
                # Get appropriate column (output variable) then use the three indices to get the correct value
                # NOTE - reversed order compared to original EOS table, see the documentation above for explanation
                new_row = [var_dict[name][k, j, i] for name in var_names]
                # Add to our array
                labels_array.append(new_row)

    # Save examples as HDF5 file
    with h5py.File(save_name, 'w') as f:
        # Save the examples under "my dataset"
        dataset = f.create_dataset('features', data=features_array)
        dataset = f.create_dataset('labels', data=labels_array)
        # Save the names of variables of examples in a separate dataset
        dataset = f.create_dataset('var_names', data=var_names)


def generate_c2p_data_tabular_eos(eos_table: h5py._hl.files.File, number_of_points: int = 10000, save_name: str = "", save_raw_data=False) -> list:
    """
    Generates training data of specified size by sampling from tabulated EOS and performing the P2C transformation.
    :param eos_table: An h5py File object containing the contents of the EOS table that we read in.
    :param number_of_points: The number of data points to be generated.
    :param save_name: In case we save the data, the name of the .csv file to which the data is saved.
    :return: list of which rows of sampled data of conserved and primitive variables.
    """

    # Initialize empty arrays for saving data
    features = []  # features for training the network
    labels = []  # labels for training the network
    if save_raw_data:
        data = []  # save raw data

    # Names for saving later on:
    var_names = ["logrho", "rho", "logeps", "eps", "v", "logtemp", "temp", "ye", "logp", "p", "logD", "D", "logS", "S", "logtau", "tau"]

    # Get the appropriate Numpy arrays from the EOS table
    ye_table   = eos_table["ye"][()]
    temp_table = eos_table["logtemp"][()]
    rho_table  = eos_table["logrho"][()]
    eps_table  = eos_table["logenergy"][()]
    p_table    = eos_table["logpress"][()]

    # Get the sizes of the table axes: the table uses the form (Y_e, T, rho)
    len_ye   = eos_table["pointsye"][()][0]
    len_temp = eos_table["pointstemp"][()][0]
    len_rho  = eos_table["pointsrho"][()][0]

    # Sample and generate a new row for the training data
    for i in range(number_of_points):
        # Sample velocity uniformly
        v = random.uniform(V_MIN, V_MAX)

        # Sample indices for the table:
        ye_index   = np.random.choice(len_ye)
        temp_index = np.random.choice(len_temp)
        rho_index  = np.random.choice(len_rho)

        # Get the values:
        ye      = ye_table[ye_index]
        logtemp = temp_table[temp_index]
        logrho  = rho_table[rho_index]
        logeps  = eps_table[ye_index, temp_index, rho_index]
        logp    = p_table[ye_index, temp_index, rho_index]

        # Get rid of the log values
        temp = 10 ** logtemp
        rho  = 10 ** logrho
        eps  = 10 ** logeps
        p    = 10 ** logp

        # Do P2C with above values
        Dvalue   = D(rho, eps, v, p)
        Svalue   = S(rho, eps, v, p)
        tauvalue = tau(rho, eps, v, p)

        # Get the log values
        logD   = np.log10(Dvalue)
        logS   = np.log10(Svalue)
        logtau = np.log10(tauvalue)

        # Add the values to a new row for the raw data
        if save_raw_data:
            new_row = [logrho, rho, logeps, eps, v, logtemp, temp, ye, logp, p, logD, Dvalue, logS, Svalue, logtau, tauvalue]
            # Append the row to the list
            data.append(new_row)

        # Save the feature and label data
        features.append([logD, logS, logtau, ye])
        # TODO - nested list or not???
        labels.append([logp])

    # In the end, we will save as HDF5
    with h5py.File(save_name, 'w') as f:
        # Save the features and labels data
        dataset = f.create_dataset("features", data=features)
        dataset = f.create_dataset("labels", data=labels)
        if save_raw_data:
            # Convert to an array
            data = np.array(data)
            # Save the raw data under "my dataset"
            dataset = f.create_dataset('var_names', data=var_names)
            for i in range(len(var_names)):
                dataset = f.create_dataset(var_names[i], data=data[:, i])

    # Don't forget to close EOS tables!
    eos_table.close()

    return None
