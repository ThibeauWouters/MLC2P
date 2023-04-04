"""This is a script that takes care of reading and processing data. Generating data is done in physics.py"""

import physics
import data
from typing import Callable
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


def read_training_data(filename: str) -> pd.DataFrame:
    """
    Read data from a given file and create a Pandas DataFrame from it.
    :param filename: Location of the data to be read.
    :return: Pandas dataframe of the data stored in the CSV.
    """

    return pd.read_csv(filename)


# def normalize(x, mean, std):
#     return (x-mean)/std


class CustomDataset(Dataset):
    """
    Custom data set used to represent the C2P training data conveniently for neural network training.
    """

    def __init__(self, all_data: pd.DataFrame, feature_names: list = ["D", "S", "tau"], label_names: list = ["p"], 
                 normalization_function: Callable = None):
        """
        Initializes the class by separating data into features and labels. See PyTorch tutorial for more information.
        :param all_data: The data of tuples of conserved and primite variables.
        """

        # Hard copy (weird stuff happening?)
        all_data = all_data.copy()

        # Get the desired input/output data out of the Pandas dataframe and as np arrays 
        feature_data = np.array([all_data[var] for var in feature_names], dtype=np.float32).transpose()
        label_data   = np.array([all_data[var] for var in label_names], dtype=np.float32).transpose()

        # Normalize input data if given a StandardScaler object's function
        if normalization_function is not None:
            feature_data = normalization_function(feature_data)

        # Convert to Torch tensors for PyTOrch
        features = torch.from_numpy(feature_data)
        labels   = torch.from_numpy(label_data)

        # for i in range(len(all_data)):
        #     # Separate the features
        #     new_feature = [all_data[var][i] for var in feature_names]
        #     features.append(torch.tensor(new_feature, dtype=torch.float32))
        #     # Separate the labels
        #     new_label = [all_data[var][i] for var in label_names]
        #     labels.append(torch.tensor(new_label, dtype=torch.float32))

        # Save as instance variables to the dataloader
        self.features = features
        self.labels   = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        """
        Gets an item from the dataset based on a specified index.
        :param idx: Index used to fetch the item.
        :return: Tuple of feature and its label.
        """
        # Get the feature, but normalized
        feature = self.features[idx]

        # Get the label
        label = self.labels[idx]
        return feature, label


def generate_CustomDataset(size: int) -> CustomDataset:
    """
    Generates a CustomDataset object of specified size. Uses ideal gas law.
    :param size: Number of data points in the data set.
    :return: CustomDataset object containing the desired amount of data points.
    """
    df = physics.generate_data_as_df(size)
    return CustomDataset(df)


def generate_dataloader(size: int) -> DataLoader:
    """
    Generates a DataLoader object of a specified size for training or testing data.
    :param size: Number of datapoints to be included in the dataloader.
    :return: A DataLoader object of the specified size.
    """
    training_data_csv = physics.generate_data_as_df(size)
    training_data = data.CustomDataset(training_data_csv)
    return DataLoader(training_data, batch_size=32)
