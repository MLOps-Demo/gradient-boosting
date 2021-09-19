"""
Standard Scaling the raw data
"""
from sklearn.preprocessing import StandardScaler
from mltrace import create_component, register
import numpy as np
import pickle


@register(
    component_name="Pre-Processing", input_vars=["filename"], output_vars=["clean_version"]
)
def normalize():
    print("Normalizing the data")

    print("Loading split data")
    x_train = np.load("../data/processed_data/x_train.npy")
    x_test = np.load("../data/processed_data/x_test.npy")
    print("done")

    print("Scaling data with Standard Scaler")
    scaling = StandardScaler()
    scaling.fit(x_train)
    print("done")

    with open("../data/scaling_model.pkl", "wb") as x_f:
        pickle.dump(scaling, x_f)


if __name__ == '__main__':
    # Create component
    create_component(
        name="Pre-Processing",
        description="Normalizes data",
        owner="shreyas",
        tags=["etl"],
    )
    normalize()
