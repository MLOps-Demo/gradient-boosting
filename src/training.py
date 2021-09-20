"""
Training Gradient boosting model with Grid Search CV
"""
from learning_curves import plot_learning_curve, feature_importance, deviance
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
import numpy as np
import pickle
import yaml
import dvclive

params = yaml.safe_load(open("params.yaml"))["training"]
n_estimators = params["n_est"]
max_depth = params["m_depth"]
learning_rate = params["lr"]
min_samples_split = params["min_split"]
min_samples_leaf = params["min_leaf"]

# param_grid = {'n_estimators': n_estimators,
#               'max_depth': max_depth,
#               'learning_rate': learning_rate,
#               'min_samples_split': min_samples_split,
#               'min_samples_leaf': min_samples_leaf}


def training():
    print("Training GBRT model")

    print("Loading scaled features and labels")
    x_train = np.load("data/processed_data/x_train.npy")
    y_train = np.load("data/processed_data/y_train.npy")
    scaling_model = pickle.load(open("data/scaling_model.pkl", "rb"))
    x_tr_scale = scaling_model.transform(x_train)
    print("done")

    model = GradientBoostingRegressor(n_estimators=n_estimators,
                                      min_samples_split=min_samples_split,
                                      max_depth=max_depth,
                                      learning_rate=learning_rate,
                                      min_samples_leaf=min_samples_leaf)
    model.fit(x_tr_scale, y_train)

    # Plot Learning Curves
    title = "Learning Curves"
    kfold = KFold(n_splits=10)
    fig1 = plot_learning_curve(model, title, x_tr_scale, y_train, cv=kfold, n_jobs=2)
    fig1.savefig("learning_curve.png")

    with open("data/gbrt_model.pkl", "wb") as x_f:
        pickle.dump(model, x_f)


if __name__ == '__main__':
    training()
