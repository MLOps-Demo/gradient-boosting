import itertools
import subprocess

# Automated grid search experiments
n_estimators = [250, 500]
max_depth = [3, 4]
learning_rate = [0.1, 0.01]
min_samples_split = [2, 4, 6]
min_samples_leaf = [1, 3, 5]

# Iterate over all combinations of hyper-parameter values.
for n_est, m_depth, lr, min_split, min_leaf in itertools.product(n_estimators, max_depth, learning_rate, min_samples_split, min_samples_leaf):
    # Execute "dvc exp run --queue --set-param train.n_est=<n_est> --set-param train.min_split=<min_split>".
    subprocess.run(["dvc", "exp", "run", "--queue",
                    "--set-param", f"training.n_est={n_est}",
                    "--set-param", f"training.m_depth={m_depth}",
                    "--set-param", f"training.lr={lr}",
                    "--set-param", f"training.min_split={min_split}",
                    "--set-param", f"training.min_leaf={min_leaf}"])
