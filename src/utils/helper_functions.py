import os.path
import os
import numpy as np
import pandas as pd
import ast
import re

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm, qmc


# -------------------------------------------------
# Parse inputs.txt
# -------------------------------------------------
def read_inputs(file_path):

    records = []
    buffer = ""

    with open(file_path) as f:
        for line in f:
            buffer += line
            if buffer.count("[") == buffer.count("]"):
                records.append(eval(buffer, {"array": np.array}))
                buffer = ""

    return records


# -------------------------------------------------
# Parse outputs.txt
# -------------------------------------------------
def read_outputs(file_path):

    records = []
    buffer = ""

    with open(file_path) as f:
        for line in f:
            buffer += line
            if buffer.count("[") == buffer.count("]"):

                cleaned = re.sub(r'np\.float64\((.*?)\)', r'\1', buffer)

                records.append(ast.literal_eval(cleaned))

                buffer = ""

    return records


# -------------------------------------------------
# Expected Improvement
# -------------------------------------------------
def expected_improvement(X, X_sample, Y_sample, model, xi=0.01):
    """
    Computes EI at points X based on existing samples.
    """
    mu, sigma = model.predict(X, return_std=True)
    mu_sample_opt = np.max(Y_sample)
    sigma = sigma.reshape(-1, 1)
    mu = mu.reshape(-1, 1)

    with np.errstate(divide='warn'):
        improvement = mu - mu_sample_opt - xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei.ravel()

# -------------------------------------------------
# Bayesian optimisation step
# -------------------------------------------------
def propose_location(X_sample, Y_sample, dim, n_candidates=20000):
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(dim),
        length_scale_bounds=(1e-2, 10),
        nu=2.5
    ) + WhiteKernel(noise_level=1e-6)
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=42
    )

    model.fit(X_sample, Y_sample)
    sampler = qmc.LatinHypercube(d=dim, seed=42)
    X_candidates = sampler.random(n=n_candidates)
    ei = expected_improvement(X_candidates, X_sample, Y_sample, model)
    best_idx = np.argmax(ei)

    return X_candidates[best_idx]

# -------------------------------------------------
# Build dataset for each function
# -------------------------------------------------
def build_dataset(function_id, input_records, output_records, initial_data_path):

    path = os.path.join(initial_data_path, f"function_{function_id}")

    initial_inputs = np.load(os.path.join(path, "initial_inputs.npy"))
    initial_outputs = np.load(os.path.join(path, "initial_outputs.npy"))

    df_initial = pd.concat(
        [pd.DataFrame(initial_inputs), pd.DataFrame(initial_outputs)],
        axis=1
    )

    input_rows = [rec[function_id-1] for rec in input_records]
    output_rows = [rec[function_id-1] for rec in output_records]

    df_addl = pd.concat(
        [pd.DataFrame(input_rows), pd.DataFrame(output_rows)],
        axis=1
    )

    df_combined = pd.concat([df_initial, df_addl], ignore_index=True)

    return df_combined


# -------------------------------------------------
# Run Bayesian optimisation
# -------------------------------------------------
def run_bayesian_optimisation(datasets, n_func):

    next_candidates = []

    for j in range(1, n_func + 1):

        df = datasets[f"function_{j}"]

        X_sample = df.iloc[:, :-1].values
        Y_sample = df.iloc[:, -1].values

        dim = X_sample.shape[1]

        x_next = propose_location(X_sample, Y_sample, dim)

        next_candidates.append(x_next)

    return next_candidates


# -------------------------------------------------
# Format output candidate
# -------------------------------------------------
def format_candidate(x):

    return "-".join([f"{min(max(xi,0),0.999999):.6f}" for xi in x])


# -------------------------------------------------
# Save submission file
# -------------------------------------------------
def save_candidates(next_candidates, file_path):

    with open(file_path, "w") as f:

        for x in next_candidates:

            line = format_candidate(x)

            f.write(line + "\n")