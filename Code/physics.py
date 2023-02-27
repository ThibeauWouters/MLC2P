import random
import csv
import data
import pandas as pd
from data import CustomDataset
# Define ranges of parameters to be sampled (see paper Section 2.1)
RHO_MIN = 0
RHO_MAX = 10.1
EPS_MIN = 0
EPS_MAX = 2.02
V_MIN = 0
V_MAX = 0.721


# Methods to compute physical quantities. See Dieselhorst et al. paper for more information.
def eos(rho: float, eps: float, gamma: float = 5/3) -> float:
    """
    See eq2 Dieselhorst et al. paper. Computes the EOS being an analytic Gamma law
    """
    return (gamma - 1) * rho * eps


def h(rho: float, eps: float, v: float) -> float:
    """
    See eq2 Dieselhorst et al. paper. Computes enthalpy
    """
    # First compute pressure
    p = eos(rho, eps)
    # Then compute enthalpy
    return 1 + eps + p/rho


def W(rho: float, eps: float, v: float) -> float:
    """
    See eq2 Dieselhorst et al. paper. Lorentz factor. Here, in 1D so v = v_x
    """
    return (1-v**2)**(-1/2)


def D(rho: float, eps: float, v: float) -> float:
    """
    See eq2 Dieselhorst et al. paper.
    """
    return rho*W(rho, eps, v)


def S(rho: float, eps: float, v: float) -> float:
    """
    See eq2 Dieselhorst et al. paper.
    """
    return rho*h(rho, eps, v)*((W(rho, eps, v))**2)*v


def tau(rho: float, eps: float, v: float) -> float:
    """
    See eq2 Dieselhorst et al. paper.
    """
    return rho*(h(rho, eps, v))*((W(rho, eps, v))**2) - eos(rho, eps) - D(rho, eps, v)


def generate_data(number_of_points: int = 10000, save: bool = False, name: str = "C2P_data") -> list:
    """
    Generates training data of specified size by sampling by performing the P2C transformation.
    :param number_of_points: The number of data points to be generated.
    :param save: Decides whether the data, after generation, gets saved or not.
    :param name: In case we save the data, the name of the .csv file to which the data is saved.
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
        p        = eos(rho, eps)
        Dvalue   = D(rho, eps, v)
        Svalue   = S(rho, eps, v)
        tauvalue = tau(rho, eps, v)

        # Add the values to a new row
        new_row = [rho, eps, v, p, Dvalue, Svalue, tauvalue]

        # Append the row to the list
        data.append(new_row)

    # Done generating data, now save if wanted by the user:
    if save:
        # Save as CSV, specify the header
        header = ['rho', 'eps', 'v', 'p', 'D', 'S', 'tau']
        # Get the correct filename, pointing to the "data" directory
        filename = 'D:/Coding/master-thesis-AI/data' + name
        if (name[-3:]) != ".csv":
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


def generate_data_as_df(number_of_points: int = 1000, save: bool = False, name: str = "C2P_data") -> pd.DataFrame:
    """
    Same as above function, but creates Pandas DataFrame out of generated data before returning.
    :param number_of_points: The number of data points to be generated.
    :param save: Decides whether the data, after generation, gets saved or not.
    :param name: In case we save the data, the name of the .csv file to which the data is saved.
    :return: Pandas DataFrame of sampled data of conserved and primitive variables.
    """
    # Generate the data using function above
    test_data_csv = generate_data(number_of_points, save=save, name=name)
    # Return after making correct Pandas DataFrame
    return pd.DataFrame(test_data_csv, columns=['rho', 'eps', 'v', 'p', 'D', 'S', 'tau'])
