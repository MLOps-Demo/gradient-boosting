from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
import numpy as np
import pickle

param_grid = {'n_estimators': [500],
              'max_depth': [3, 4],
              'learning_rate': [0.1, 0.01, 0.001],
              'min_samples_split': [2, 4, 6],
              'min_samples_leaf': [1, 3, 5]}


def training():
    print("Training GBRT model with Grid Search")
    model = GradientBoostingRegressor()

    print("Loading scaled features and labels")
    x_train = np.load("data/processed_data/x_train.npy")
    y_train = np.load("data/processed_data/y_train.npy")
    scaling_model = pickle.load(open("data/scaling_model.pkl", "rb"))
    x_tr_scale = scaling_model.transform(x_train)
    print("done")

    print("Cross Validation Started")
    kfold = KFold(n_splits=10)
    grid_search = GridSearchCV(model, param_grid, cv=kfold, scoring = 'neg_mean_squared_error')
    grid_search.fit(x_tr_scale, y_train)
    print("done")

    with open("data/gbrt_model.pkl", "wb") as x_f:
        pickle.dump(grid_search, x_f)


if __name__ == '__main__':
    training()
