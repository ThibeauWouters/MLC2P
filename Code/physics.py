"""
This script is dedicated to generating the data samples used for training and testing the neural networks.
"""
import random
import csv
import data
import pandas as pd
import numpy as np
import torch
# from data import CustomDataset
import os
import h5py

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


def h(rho: float, eps: float, v: float, p=None) -> float:
    """
    See eq2 Dieselhorst et al. paper. Computes enthalpy
    """
    # First compute pressure
    p = ideal_eos(rho, eps)
    # Then compute enthalpy
    return 1 + eps + p/rho


def W(rho: float, eps: float, v: float, p: float = None) -> float:
    """
    See eq2 Dieselhorst et al. paper. Lorentz factor. Here, in 1D so v = v_x
    """
    return (1-v**2)**(-1/2)


def D(rho: float, eps: float, v: float, p: float = None) -> float:
    """
    See eq2 Dieselhorst et al. paper.
    """
    return rho*W(rho, eps, v)


def S(rho: float, eps: float, v: float, p: float = None) -> float:
    """
    See eq2 Dieselhorst et al. paper.
    """
    return rho*h(rho, eps, v)*((W(rho, eps, v))**2)*v


def tau(rho: float, eps: float, v: float, p: float = None) -> float:
    """
    See eq2 Dieselhorst et al. paper.
    """
    # If the pressure is None, rely on the ideal gas EOS
    if p is None:
        p = ideal_eos(rho, eps)
    return rho*(h(rho, eps, v))*((W(rho, eps, v))**2) - p - D(rho, eps, v)


def p2c(rho: float, eps: float, v: float, p: float):
    """
    Performs the P2C transformation, as given by the Eq. 2 of Dieselhorst et al. paper.
    """
    # If the pressure is None, rely on the ideal gas EOS
    if p is None:
        p = ideal_eos(rho, eps)

    return D(rho, eps, v, p), S(rho, eps, v, p), tau(rho, eps, v, p)


def get_prims(D_value, S_value, tau_value, p):

    v = S_value/(tau_value + D_value + p)
    if isinstance(v, torch.Tensor):
        W_value = 1/torch.sqrt(1-v**2)
    else:
        W_value = 1/np.sqrt(1-v**2)
    eps_value = (tau_value + D_value*(1-W_value) + p*(1-W_value**2))/(D_value*W_value)
    rho_value = D_value/W_value

    return v, W_value, eps_value, rho_value

#############################
# DATA GENERATING FUNCTIONS #
#############################


def generate_data(number_of_points: int = 10000, save_name: str = "") -> list:
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
        rho = random.uniform(RHO_MIN, RHO_MAX)
        eps = random.uniform(EPS_MIN, EPS_MAX)
        v   = random.uniform(V_MIN, V_MAX)

        # Use above transformations to compute the pressure and conserved variables D, S, tau
        # Compute the pressure using an ideal gas law
        p        = ideal_eos(rho, eps)
        # Do the P2C transformation
        Dvalue, Svalue, tauvalue = p2c(rho, eps, v, p)

        # Add the values to a new row
        new_row = [rho, eps, v, p, Dvalue, Svalue, tauvalue]

        # Append the row to the list
        data.append(new_row)

    # Done generating data, now save if wanted by the user:
    if len(save_name) > 0:
        # Save as CSV, specify the header
        header = ['rho', 'eps', 'v', 'p', 'D', 'S', 'tau']
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
    return data


def generate_data_as_df(number_of_points: int = 1000, save_name: str = "") -> pd.DataFrame:
    """
    Same as above function, but creates Pandas DataFrame out of generated data before returning.
    :param number_of_points: The number of data points to be generated.
    :param save_name: In case we save the data, the name of the .csv file to which the data is saved.
    :return: Pandas DataFrame of sampled data of conserved and primitive variables.
    """
    # Generate the data using function above
    test_data_csv = generate_data(number_of_points, save_name=save_name)
    # Return after making correct Pandas DataFrame
    return pd.DataFrame(test_data_csv, columns=['rho', 'eps', 'v', 'p', 'D', 'S', 'tau'])


#################
# TABULATED EOS #
#################


def read_eos_table(filename: str):

    # Convert to proper filepath towards eos tables
    filename = os.path.join("eos_tables", filename)
    # Append it to the master dir
    filename = os.path.join(master_dir, filename)

    return h5py.File(filename, 'r')


def generate_tabular_data(eos_table: h5py._hl.files.File, number_of_points: int = 10000, save_name: str = "") -> list:
    """
    Generates training data of specified size by sampling from tabulated EOS and performing the P2C transformation.
    :param eos_table: An h5py File object containing the contents of the EOS table that we read in.
    :param number_of_points: The number of data points to be generated.
    :param save_name: In case we save the data, the name of the .csv file to which the data is saved.
    :return: list of which rows of sampled data of conserved and primitive variables.
    """

    # Initialize empty data
    data = []

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
        ye_index    = np.random.choice(len_ye)
        temp_index  = np.random.choice(len_temp)
        rho_index   = np.random.choice(len_rho)

        # Get the values:
        ye   = ye_table[ye_index]
        temp = temp_table[temp_index]
        rho  = rho_table[rho_index]
        eps  = eps_table[ye_index, temp_index, rho_index]
        p    = p_table[ye_index, temp_index, rho_index]

        # Do P2C with above values
        Dvalue   = D(rho, eps, v)
        Svalue   = S(rho, eps, v)
        tauvalue = tau(rho, eps, v)

        # Add the values to a new row
        new_row = [rho, eps, v, temp, ye, p, Dvalue, Svalue, tauvalue]

        # Append the row to the list
        data.append(new_row)

    # Done generating data, now save if wanted by the user:
    if len(save_name) > 0:
        # Save as CSV, specify the header
        header = ["rho", "eps", "v", "temp", "ye", "p", "D", "S", "tau"]
        # Get the correct filename, pointing to the "data" directory
        save_name = "Data/" + save_name
        filename = os.path.join(master_dir, save_name)
        if (save_name[-3:]) != ".csv":
            # Make sure the file extension is csv
            filename += ".csv"

        print("Saving to ", filename)

        # Open the filename and write the data to it
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            # write header
            writer.writerow(header)
            # write data
            writer.writerows(data)

    # Also return the data to the user
    return data

# def generate_data_as_df(number_of_points: int = 1000, save: bool = False, name: str = "C2P_data") -> pd.DataFrame:
#     """
#     Same as above function, but creates Pandas DataFrame out of generated data before returning.
#     :param number_of_points: The number of data points to be generated.
#     :param save: Decides whether the data, after generation, gets saved or not.
#     :param name: In case we save the data, the name of the .csv file to which the data is saved.
#     :return: Pandas DataFrame of sampled data of conserved and primitive variables.
#     """
#     # Generate the data using function above
#     test_data_csv = generate_data(number_of_points, save=save, name=name)
#     # Return after making correct Pandas DataFrame
#     return pd.DataFrame(test_data_csv, columns=['rho', 'eps', 'v', 'p', 'D', 'S', 'tau'])
