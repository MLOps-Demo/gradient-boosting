from sklearn.preprocessing import StandardScaler
import numpy as np
import dill


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

    with open("data/scaling_model.dill", "wb") as x_f:
        dill.dump(scaling, x_f)


if __name__ == '__main__':
    normalize()
