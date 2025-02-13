# -*- coding: utf-8 -*-
"""
# **This file contains all suggorates used for pre-training VQE with large-scale TFIM models**
"""
import os
import json
from numpy import linalg as LA
from scipy.sparse.linalg import eigsh
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from itertools import combinations, product
import optax
import jax
from sklearn.neural_network import MLPRegressor
from joblib import dump, load

 
 
n_qubits = 50
n_layers = 1  # Number of layers in the ansatz
T = 100  # Number of snapshots for shadow dataset
n_samples = 2000  # Number of data samples to generate
J_list = [0.2]  # Coupling strength
h_list = [0.5] # Field strength

"""# Surrogates Pretraining [data loading]"""

# Dataset Loading (ideal and shadow)
def load_dataset(file_path, J_, h_, idea_):
    if idea_:
        file_name = file_path + f"ideal_results_J={J_}.json"
        with open(file_name, 'r') as file:
            data = json.load(file)
    else:
        file_name = file_path + f"labeled_data_shadow_J={J_}.json"
        with open(file_name, 'r') as file:
            data = json.load(file)
    return data

# Build Train and test dataset
num_train = 1500
num_test = 500

dataset_file_path = 'Dataset_Pretraining/large_scale_qubits/'

# Shadow data in

# Metrics

def evaluate_predictions(predictions, test_results):
    """
    Evaluate the relation between predictions and test results using R^2
    and Pearson correlation coefficient.

    Args:
    - predictions (numpy.ndarray): Predicted values.
    - test_results (numpy.ndarray): True test results.

    Returns:
    - dict: A dictionary containing R^2 and Pearson correlation coefficient.
    """
    # Compute R^2 score
    r2 = r2_score(test_results, predictions)

    # Compute Pearson correlation coefficient
    pearson_corr, _ = pearsonr(test_results, predictions)

    return {"R^2": r2, "Pearson Correlation": pearson_corr}

"""# Proposed ML Surrogates"""

def generate_frequency_vectors(input_dim, Lambda):
    """
    Generate frequency vectors with truncated Hamming weight up to Lambda.

    Args:
        input_dim (int): Dimensionality of the input data (number of features).
        Lambda (int): Maximum Hamming weight for the truncated frequency set.

    Returns:
        numpy.ndarray: Frequency matrix of shape (n_vectors, input_dim),
                       where n_vectors is determined by input_dim and Lambda.
    """
    frequency_vectors = []
    frequency_vectors.append(np.zeros(input_dim)) # frequence vector with zero Hamming weight
    for hamming_weight in range(1, Lambda + 1):
        for indices in combinations(range(input_dim), hamming_weight):
            for signs in product([-1, 1], repeat=hamming_weight):
                freq_vector = np.zeros(input_dim)
                for idx, sign in zip(indices, signs):
                    freq_vector[idx] = sign
                frequency_vectors.append(freq_vector)

    return np.array(frequency_vectors)

def feature_map_deterministic(data, Lambda):
    """
    Compute a deterministic feature map based on truncated Hamming weight.

    Args:
        data (numpy.ndarray): Input matrix of shape (n_samples, input_dim).
        Lambda (int): Maximum Hamming weight for the truncated frequency set.

    Returns:
        numpy.ndarray: Feature map matrix of shape (n_samples, n_vectors),
                       where n_vectors is determined by input_dim and Lambda.
    """
    n_samples, input_dim = data.shape

    # Generate deterministic frequency vectors
    sampled_W = generate_frequency_vectors(input_dim, Lambda)  # Shape: (n_vectors, input_dim)

    # Expand data and sampled_W to align dimensions
    data_expanded = np.expand_dims(data, axis=1)   
    W_expanded = np.expand_dims(sampled_W, axis=0)   

    f_x_w = np.where(
        W_expanded == 0,  # If w_j == 0
        1,
        np.where(W_expanded == 1, np.cos(data_expanded), np.sin(data_expanded))
    )   

    # Compute the product across dimensions to get g(x; w)
    features = np.prod(f_x_w, axis=2)   

    # Weight features by 2^||w||_0
    hamming_weights = np.count_nonzero(sampled_W, axis=1)   
    weight_factors = 2 ** hamming_weights   
    features *= weight_factors   

    return features

##################################
## Implementation of Surrogate ###
##################################


# Generate deterministic feature map by using the ideal_dataset and shadow_dataset
def ML_predictor(J_, h_, idea_setting_, num_train_, num_test_, Lambda_, results_file_path, model_file_path_):

    if idea_setting_ is True:
        data_all = load_dataset(dataset_file_path, J_, h_, True)

        train_data =  np.array(data_all['para'])[:num_train_]
        train_data = np.reshape(train_data, newshape = (num_train_, -1))
        train_labels = np.array(data_all['labels'])[:num_train_]

        test_data = np.array(data_all['para'])[-num_test_:]
        test_data = np.reshape(test_data, newshape = (num_test_, -1))
        test_labels = np.array(data_all['labels'])[-num_test_:]
    else:
        data_all = load_dataset(dataset_file_path, J_, h_, False)
        train_data =  np.array(data_all['parameters'])[:num_train_]
        train_data = np.reshape(train_data, newshape = (num_train_, -1))
        train_labels = np.array(data_all['shadow_labels'])[:num_train_]
        train_labels  = np.squeeze(train_labels)

        test_data = np.array(data_all['parameters'])[-num_test_:]
        test_data = np.reshape(test_data, newshape = (num_test_, -1))
        test_labels = np.array(data_all['shadow_labels'])[-num_test_:]
        test_labels  = np.squeeze(test_labels)

    # Feature map implementation
    feature_map_trunc = feature_map_deterministic(train_data, Lambda_)
    print("Feature Map Shape:", feature_map_trunc.shape)
    # Model training
    model_Trunc = Ridge(alpha=1)
    model_Trunc.fit(feature_map_trunc, train_labels)

    # Predict on new transformed data
    phi_new = feature_map_deterministic(test_data, Lambda_)
    predictions = model_Trunc.predict(phi_new)

    # Evaluate
    metrics = evaluate_predictions(predictions, test_labels)
    mse = mean_squared_error(test_labels, predictions)
    print(f"Mean Squared Error: {mse}")
    print("Evaluation Metrics:")
    print(f"R^2: {metrics['R^2']:.4f}")
    print(f"Pearson Correlation: {metrics['Pearson Correlation']:.4f}")

    # Prepare results for JSON
    result = {
        "J": J_,
        "h": h_,
        "mse": mse,
        "metrics": {
            "R^2": metrics["R^2"],
            "Pearson Correlation": metrics["Pearson Correlation"]
        }
    }

    # Save results to JSON
    try:
        with open(results_file_path, "r") as f:
            all_results = json.load(f)
    except FileNotFoundError:
        all_results = {}

 
    key = f"J={J_}_h={h_}"
    all_results[key] = result

     with open(results_file_path, "w") as f:
        json.dump(all_results, f, indent=4)

     
    dump(model_Trunc, model_file_path_)
    print(f"Model saved to {model_file_path_}")
    return model_Trunc


"""Start model training"""

# Simulation results for ideal labels

# para settings
input_dim = n_layers * (2*n_qubits -1)
Lambda = 2  # Maximum Hamming weight, hyper-parameter
results_file_path = 'ML_fitting_res_Ideal.json'
idea_setting = True


# Different TFIM Hamiltonians
for j_ in J_list:
    for h_ in h_list:
        print(f"J={j_}, h={h_}")
        model_file_path = 'ML_fitting_model_Ideal_J='+str(j_)+'.joblib'
        trained_model = ML_predictor(j_, h_, idea_setting, num_train, num_test, Lambda, results_file_path, model_file_path)

"""Shadow training"""

# Shadow models
# Simulation results for ideal labels

# para settings
input_dim = n_layers * (2*n_qubits -1)
Lambda = 2  # Maximum Hamming weight, hyper-parameter
results_file_path = 'ML_fitting_res_Shadow.json'
idea_setting = False


# Different TFIM Hamiltonians
for j_ in J_list:
    for h_ in h_list:
        print(f"J={j_}, h={h_}")
        model_file_path = 'ML_fitting_model_Shadow_J='+str(j_)+'.joblib'
        trained_model = ML_predictor(j_, h_, idea_setting, num_train, num_test, Lambda, results_file_path, model_file_path)



"""# **The second part is finding the minimum of the trained model**"""


"""Code implementation for Adam"""

# Compute gradients
def compute_gradient_auto_grad_adam(predictor, feature_map, params, Lambda, sampled_W):
    """
    Compute gradients of the predictor function with respect to circuit parameters.

    Args:
        predictor (object): Trained regression model (e.g., Ridge from sklearn).
        feature_map (callable): Function to compute features.
        params (numpy.ndarray): Circuit parameters of shape (input_dim,).
        Lambda (int): Number of Lambda.
        sampled_W (numpy.ndarray): Frequency matrix for the feature map.

    Returns:
        numpy.ndarray: Gradient vector of the same shape as params.
    """
    input_dim = params.shape[-1]

    # Step 1: Compute feature map
    features = feature_map(params.reshape(1, -1), Lambda)  # Shape: (1, num_features)

    # Step 2: Extract predictor weights
    if hasattr(predictor, "coef_"):
        weights = predictor.coef_.flatten()  
    else:
        raise ValueError("Predictor must have 'coef_' attribute.")

    # Step 3: Compute the derivative of the feature map
    params_expanded = params.reshape(1, -1)  

    # Derivative based on trigonometric features
    feature_map_derivative = np.where(
        sampled_W == 0,
        0,  # If W == 0 -> derivative = 0
        np.where(sampled_W == 1, -np.sin(params_expanded), np.cos(params_expanded))
    ) 

    # Step 4: Combine derivatives with predictor weights
    gradient = np.einsum('j,j...->...', weights, feature_map_derivative)

    return gradient


# Optimizer function
def optimize_with_adam(predictor, feature_map, initial_params, Lambda, sampled_W, learning_rate=0.01, max_iterations=500):
    """
    Optimize circuit parameters using Adam to minimize predictor output.

    Args:
        predictor (object): Trained regression model (e.g., Ridge).
        feature_map (callable): Feature mapping function.
        initial_params (np.ndarray): Initial circuit parameters.
        Lambda (int): Number of Lambda.
        sampled_W (np.ndarray): Frequency matrix.
        learning_rate (float): Adam learning rate.
        max_iterations (int): Number of optimization steps.

    Returns:
        np.ndarray: Optimized parameters.
    """
    params = np.copy(initial_params)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    loss_history = []
    vqe_energy_history = []

    for iteration in range(max_iterations):

        # Print progress every certain steps
        if iteration % 20 == 0 or iteration == max_iterations - 1:
            features = feature_map(params.reshape(1, -1), Lambda)
            current_cost = predictor.predict(features)[0]
            param_numpy = jax.device_get(params)
            loss_history.append(current_cost)
            vqe_energy_history.append({
                "iteration": iteration,
                "parameters": param_numpy.tolist(),  
            })


        gradients = compute_gradient_auto_grad_adam(predictor, feature_map, params, Lambda, sampled_W)

        updates, opt_state = optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)

    return params, loss_history, vqe_energy_history


# Simulation results for ideal and shadow labels

# para settings
input_dim = n_layers * (2*n_qubits -1)
Lambda = 2  # Maximum Hamming weight, hyper-parameter
idea_setting = False # True for ideal setting
repeat_times = 5

pop_size = 100
n_generations = 50

# Different TFIM Hamiltonians
for j_ in J_list:
    for h_ in h_list:
        print(f"J={j_}, h={h_}")
        model_file_path = 'ML_fitting_model_Shadow_J='+str(j_)+'.joblib'
        loaded_trained_model = load(model_file_path)
        print("Model loaded successfully")
        # ########################################################
        # ### Minimization of surrogate loss by Adam optimizer ###
        # ########################################################
        for iter_ in range(repeat_times):
            initial_params = np.random.uniform(-np.pi, np.pi, size=(n_layers, 2 * n_qubits - 1))
            frequency_set = generate_frequency_vectors(input_dim, Lambda)

            results_file_path = "Adam_Optimization_Results.json"

            optimized_params, loss_history, vqe_energy_history = optimize_with_adam(
                predictor=loaded_trained_model,  # Trained Ridge model
                feature_map=feature_map_deterministic,
                initial_params=initial_params,
                Lambda=Lambda,
                sampled_W=frequency_set,
                learning_rate=0.008,
                max_iterations=501
            )

            Inter_valide_output_file = 'Adam_Shadow_J='+str(j_)+'Repeat'+str(iter_)+'.json'
            vqe_energy_history.append({
                "iteration": 0,
                "parameters": initial_params.tolist(),  # Convert numpy array to list for JSON compatibility
            })
            vqe_energy_history.append({
                "iteration": 501,
                "parameters": optimized_params.tolist(),  # Convert numpy array to list for JSON compatibility
            })
            with open(Inter_valide_output_file, "w") as f:
                json.dump(vqe_energy_history, f, indent=4)

            print(f"VQE energy history saved to {Inter_valide_output_file}")

            # Prepare results for JSON
            result = {
                "J": j_,
                "h": h_,
                "iteration": iter_,
                "loss_history": loss_history,
            }

            try:
                with open(results_file_path, "r") as f:
                    all_results = json.load(f)
            except FileNotFoundError:
                all_results = {}

            key = f"J={j_}_h={h_}_iter={iter_}"
            all_results[key] = result

            with open(results_file_path, "w") as f:
                json.dump(all_results, f, indent=4)
 

 