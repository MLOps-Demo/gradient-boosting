"""
Standard Scaling the raw data
"""
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle


def normalize():
    print("Normalizing the data")

    print("Loading split data")
    x_train = np.load("data/processed_data/x_train.npy")
    x_test = np.load("data/processed_data/x_test.npy")
    print("done")

    print("Scaling data with Standard Scaler")
    scaling = StandardScaler()
    scaling.fit(x_train)
    print("done")

    with open("data/scaling_model.pkl", "wb") as x_f:
        pickle.dump(scaling, x_f)


if __name__ == '__main__':
    normalize()
