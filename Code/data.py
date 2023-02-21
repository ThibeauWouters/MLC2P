"""This is a script that takes care of reading and processing data. Generating data is done in physics.py"""

import pandas as pd
import torch
from torch.utils.data import Dataset


def read_training_data(filename: str) -> pd.DataFrame:
    """
    Reads data from a CSV.
    Args:
        filename (str): Location of the data to be read.
    Returns:
        pd.DataFrame: Pandas dataframe of the data stored in the CSV.
    """

    return pd.read_csv(filename)


class CustomDataset(Dataset):
    """
    Custom data set used to represent the C2P training data conveniently for neural network training.
    """

    def __init__(self, all_data: pd.DataFrame, transform=None, target_transform=None):
        """
        Initializes the class by separating data into features and labels. See PyTorch tutorial for more information.
        :param all_data: The data of tuples of conserved and primite variables.
        :param transform: Don't know
        :param target_transform: Don't know
        """

        # TODO - remove this? Not sure what it does
        self.transform        = transform
        self.target_transform = target_transform

        # Separate features (D, S, tau) from the labels (p). Initialize empty lists
        features = []
        labels   = []

        for i in range(len(all_data)):
            # Separate the features
            new_feature = [all_data['D'][i], all_data['S'][i], all_data['tau'][i]]
            features.append(torch.tensor(new_feature, dtype=torch.float32))
            # Separate the labels
            new_label = [all_data['p'][i]]
            labels.append(torch.tensor(new_label, dtype=torch.float32))

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
        # Get the feature
        feature = self.features[idx]
        if self.transform:
            feature = self.transform(feature)
        # Get the label
        label = self.labels[idx]
        if self.target_transform:
            feature = self.target_transform(label)
        return feature, label
