import sklearn.metrics as metrics
import pickle
import json
import numpy as np


def evaluate():
    print("Model Evaluation")
    x_test = np.load("data/processed_data/x_test.npy")
    y_test = np.load("data/processed_data/y_test.npy")

    scaling_model = pickle.load(open("data/scaling_model.pkl", "rb"))
    x_te_scale = scaling_model.transform(x_test)
    print("done")

    model = pickle.load(open("data/gbrt_model.pkl", "rb"))
    predictions = model.predict(x_te_scale)

    mse = metrics.mean_squared_error(y_test, predictions)
    r2 = metrics.r2_score(y_test, predictions)

    with open("scores.json", "w") as fd:
        json.dump({"mse": mse, "r2": r2}, fd, indent=4)


if __name__ == '__main__':
    evaluate()
