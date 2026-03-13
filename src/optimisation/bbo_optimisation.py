from src.utils.helper_functions import *
# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    n_week = 3
    n_func = 8

    data_path = r'D:\Surajit\MachineLearning_ImperialCollege\bbo-capstone\data'
    initial_data_path = os.path.join(data_path, "initial_data")
    addl_data_path = os.path.join(data_path, f"additional_data_week{n_week}")
    inputs_file = os.path.join(addl_data_path, "inputs.txt")
    outputs_file = os.path.join(addl_data_path, "outputs.txt")
    input_records = read_inputs(inputs_file)
    output_records = read_outputs(outputs_file)
    datasets = {}

    for j in range(1, n_func + 1):

        datasets[f"function_{j}"] = build_dataset(
            j,
            input_records,
            output_records,
            initial_data_path
        )

    next_candidates = run_bayesian_optimisation(datasets, n_func)
    submission_file = os.path.join(data_path, "next_points.txt")
    save_candidates(next_candidates, submission_file)
    print("Next candidate points saved to:", submission_file)












# -----------------------------
# Expected Improvement
# -----------------------------
# def expected_improvement(X, X_sample, Y_sample, model, xi=0.01):
#     """
#     Computes EI at points X based on existing samples.
#     """
#     mu, sigma = model.predict(X, return_std=True)
#     mu_sample_opt = np.max(Y_sample)
#
#     sigma = sigma.reshape(-1, 1)
#     mu = mu.reshape(-1, 1)
#
#     with np.errstate(divide='warn'):
#         improvement = mu - mu_sample_opt - xi
#         Z = improvement / sigma
#         ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
#         ei[sigma == 0.0] = 0.0
#
#     return ei.ravel()

# -----------------------------
# Bayesian optimisation step
# -----------------------------
# def propose_location(X_sample, Y_sample, dim, n_candidates=20000):
#     """
#     Propose next point using Expected Improvement.
#     Uses Latin Hypercube sampling for better coverage.
#     """
#     # -----------------------------
#     # Gaussian Process
#     # -----------------------------
#     kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
#         length_scale=np.ones(dim),
#         length_scale_bounds=(1e-2, 10),
#         nu=2.5
#     ) + WhiteKernel(noise_level=1e-6)
#
#     model = GaussianProcessRegressor(
#         kernel=kernel,
#         normalize_y=True,
#         n_restarts_optimizer=5,
#         random_state=42
#     )
#
#     model.fit(X_sample, Y_sample)
#
#     # -----------------------------
#     # Latin Hypercube candidate sampling
#     # -----------------------------
#     sampler = qmc.LatinHypercube(d=dim, seed=42)
#     X_candidates = sampler.random(n=n_candidates)
#
#     # -----------------------------
#     # Evaluate Expected Improvement
#     # -----------------------------
#     ei = expected_improvement(X_candidates, X_sample, Y_sample, model)
#
#     best_idx = np.argmax(ei)
#
#     return X_candidates[best_idx]

# -----------------------------
# Run for each function
# -----------------------------
#
#
# def read_inputs(file_path):
#     records = []
#     buffer = ""
#
#     with open(file_path) as f:
#         for line in f:
#             buffer += line
#             if buffer.count("[") == buffer.count("]"):
#                 records.append(eval(buffer, {"array": np.array}))
#                 buffer = ""
#
#     return records
#
#
# def read_outputs(file_path):
#     records = []
#     buffer = ""
#
#     with open(file_path) as f:
#         for line in f:
#             buffer += line
#             if buffer.count("[") == buffer.count("]"):
#                 cleaned = re.sub(r'np\.float64\((.*?)\)', r'\1', buffer)
#                 records.append(ast.literal_eval(cleaned))
#                 buffer = ""
#
#     return records


# def build_dataset(function_id, input_records, output_records, initial_data_path):
#
#     path = os.path.join(initial_data_path, f"function_{function_id}")
#
#     initial_inputs = np.load(os.path.join(path, "initial_inputs.npy"))
#     initial_outputs = np.load(os.path.join(path, "initial_outputs.npy"))
#
#     df_initial = pd.concat(
#         [pd.DataFrame(initial_inputs), pd.DataFrame(initial_outputs)],
#         axis=1
#     )
#
#     input_rows = [rec[function_id-1] for rec in input_records]
#     output_rows = [rec[function_id-1] for rec in output_records]
#
#     df_addl = pd.concat(
#         [pd.DataFrame(input_rows), pd.DataFrame(output_rows)],
#         axis=1
#     )
#
#     df_combined = pd.concat([df_initial, df_addl], ignore_index=True)
#
#     return df_combined
#
#
# if __name__ == "__main__":
#
#     n_week = 3
#     n_func = 8
#
#     data_path = r'D:\Surajit\MachineLearning_ImperialCollege\bbo-capstone\data'
#     initial_data_path = os.path.join(data_path, "initial_data")
#     addl_data_path = os.path.join(data_path, f"additional_data_week{n_week}")
#
#     inputs_file = os.path.join(addl_data_path, "inputs.txt")
#     outputs_file = os.path.join(addl_data_path, "outputs.txt")
#
#     input_records = read_inputs(inputs_file)
#     output_records = read_outputs(outputs_file)
#
#     datasets = {}
#
#     for j in range(1, n_func+1):
#         datasets[f"function_{j}"] = build_dataset(
#             j,
#             input_records,
#             output_records,
#             initial_data_path
#         )





# def propose_location(X_sample, Y_sample, dim, n_candidates=20000):
#     """
#     Random-search optimisation of EI over [0,1]^dim
#     """
#     # GP kernel
#     kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-6)
#
#     model = GaussianProcessRegressor(
#         kernel=kernel,
#         normalize_y=True,
#         n_restarts_optimizer=10,
#         random_state=42
#     )
#     model.fit(X_sample, Y_sample)
#
#     # Sample candidate points uniformly
#     X_candidates = np.random.rand(n_candidates, dim)
#
#     ei = expected_improvement(X_candidates, X_sample, Y_sample, model)
#     best_idx = np.argmax(ei)
#
#     return X_candidates[best_idx]




# if __name__ == "__main__":
#     n_week = 3
#     n_func = 8
#     input_records = []
#     output_records = []
#
#     data_path = r'D:\Surajit\MachineLearning_ImperialCollege\bbo-capstone\data'
#     initial_data_path = os.path.join(data_path, 'initial_data')
#     addl_data_path = os.path.join(data_path, f"additional_data_week{n_week}")
#
#     with (open(os.path.join(addl_data_path, 'inputs.txt'), "r") as fin,
#           open(os.path.join(addl_data_path, 'outputs.txt'), "r") as fout):
#         input_buffer = ""
#         for line in fin:
#             input_buffer += line
#             if input_buffer.count("[") == input_buffer.count("]"):
#                 input_records.append(eval(input_buffer, {"array": np.array}))
#                 input_buffer = ""
#         output_buffer = ""
#         for line in fout:
#             output_buffer += line
#             if output_buffer.count("[") == output_buffer.count("]"):
#                 cleaned = re.sub(r'np\.float64\((.*?)\)', r'\1', output_buffer)
#                 output_records.append(ast.literal_eval(cleaned))
#                 output_buffer = ""
#
#     for j in range(1, n_func+1):
#
#         input_rows = [rec[j-1] for rec in input_records]
#         output_rows = [rec[j-1] for rec in output_records]
#
#         path = os.path.join(initial_data_path, f"function_{j}")
#
#         initial_inputs = np.load(os.path.join(path, 'initial_inputs.npy'))
#         initial_outputs = np.load(os.path.join(path, 'initial_outputs.npy'))
#
#         df_initial = pd.concat([
#             pd.DataFrame(initial_inputs),
#             pd.DataFrame(initial_outputs)
#         ], axis=1)
#
#         df_addl = pd.concat([
#             pd.DataFrame(input_rows),
#             pd.DataFrame(output_rows)
#         ], axis=1)
#
#         df_combined = pd.concat([df_initial, df_addl], ignore_index=True)
