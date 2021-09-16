"""
Split raw data into training and testing data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

params = yaml.safe_load(open("params.yaml"))["data-split"]
split = params["split"]
seed = params["seed"]


def data_split():
    print("Loading data from given folder")
    df = pd.read_csv('data/raw_data/clean_data.csv').set_index('NewDateTime')
    print("done")

    array = df.values

    x = array[:, :-1]
    y = array[:, -1]

    print("Splitting data into train and test")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split, random_state=seed)
    print("done")

    np.save('data/processed_data/x_train', x_train)
    np.save('data/processed_data/x_test', x_test)
    np.save('data/processed_data/y_train', y_train)
    np.save('data/processed_data/y_test', y_test)
    print("Saved data into processed_data folder")


if __name__ == '__main__':
    data_split()
