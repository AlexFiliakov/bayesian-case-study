import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import psutil
from scipy.stats import binom, nbinom, poisson
from sklearn.model_selection import KFold
from sklearn.utils import resample

from basic_distributions import (fit_beta_binom, fit_binom, fit_nb, fit_poi,
                                 fit_uniform, neg_log_likelihood_binom,
                                 neg_log_likelihood_nb)


def make_serializable(obj):
    """
    Mainly used to output Bootstrap results to file.
    Converts NumPy arrays to lists and NumPy numbers to native Python types for JSON serialization.

    Parameters:
    - obj: Object to convert to a list.

    Returns:
    - obj: Object with NumPy arrays converted to lists.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    else:
        return obj

def calc_BIC(log_likelihood, k, n):
    """
    Calculate the Bayesian Information Criterion (BIC) from the log-likelihood, number of parameters, and number of data points.
    
    Parameters:
    - log_likelihood: Log-likelihood of the model.
    - k: Number of parameters in the model.
    - n: Number of data points used to fit the model.
    
    Returns:
    - bic: Bayesian Information Criterion (BIC) value.
    """
    bic = k * np.log(n) - 2 * log_likelihood
    return bic

def bma_using_bic(model_names, param_counts, bic_values, pmf_values, train_data, test_data):
    """
    Perform Bayesian Model Averaging using BIC values and model probabilities.

    Parameters:
    model_names (list): List of model names.
    param_counts (list): List of the number of parameters for each model.
    bic_values (ndarray): Array of BIC values for each model.
    pmf_values (list): List of PMF values for each model.
    train_data (list): List of training data values.
    test_data (list): List of test data values.

    Returns:
    weights (ndarray): Array of weights for each model.
    average_pmf (ndarray): Weighted average of the PMF values.
    log_likelihood_bma (float): Log-likelihood of the averaged model.
    bic_averaged_model (float): BIC of the averaged model.
    """
    # Calculate the weights for each model based on their BIC values
    min_bic = np.min(bic_values)
    delta_bic = bic_values - min_bic
    weights = np.exp(-0.5 * delta_bic) / np.sum(np.exp(-0.5 * delta_bic))

    # Print the weights for each model
    logging.info("### BMA Weights: ###")
    for i, name in enumerate(model_names):
        logging.info(f" - {name} = {int(round(weights[i], 2)*100)}%")

    # Calculate the weighted average of the models
    average_pmf = np.zeros_like(pmf_values[0])
    for i, pmf in enumerate(pmf_values):
        average_pmf += weights[i] * np.array(pmf)

    # Calculate the log-likelihood of the averaged model on test data
    log_likelihood_bma = np.sum(np.log(average_pmf[test_data]))

    # Calculate the weighted average of the number of parameters
    num_parameters = np.sum(weights * param_counts)

    n = len(train_data) # Number of data points used to train the models

    # Calculate BIC
    bic_averaged_model = calc_BIC(log_likelihood_bma, num_parameters, n)

    logging.info(f"  Number of parameters in the averaged model: {round(num_parameters,1)}")
    logging.info(f"  BIC of the averaged model: {int(bic_averaged_model)}")

    return weights, average_pmf, log_likelihood_bma, num_parameters, bic_averaged_model

def model_func(train_data, test_data, x):
    """
    Perform the model training.
    
    Parameters:
    - train_data: Training data.
    - test_data: Testing data.
    - x: Values for the PMF.
    
    Returns:
    - weights: Weights of the component models within the BMA model.
        - The weight given to each underlying model fitted.
    - average_pmf: PMF of the BMA model.
        - The Probability Mass Function for each discrete point on the x-axis, starting from 0.
    - log_likelihood_bma: The overall log-likelihood of the resulting BMA model.
    - num_params_bma: The number of parameters of the resulting BMA model.
    - bic_averaged_model: The Bayesian Information Criterion (BIC) of the resulting BMA model.
    - params_used: The parameters used in each model.
        Description of params_used in bootstrap results:
        - Each bootstrap resample comes with the list of fitted parameters used in all of its cross-validation folds
        - At the core, params_used is a list of nested lists containing the fitted parameters in an organized way
        - Each top-line list in params_used corresponds to a cross-validation fold
        - Each fold, in turn, contains a list of the fitted base models. At the time of this writing, there were 5 base models:
            - Uniform: fitted parameter is n (maximal value)
            - Poisson: fitted parameter is λ (mean)
            - Binomial: fitted parameters are n (number of trials) and p (probability)
            - Negative Binomial: fitted parameters are r (number of claims until stop) and p (probability of a claim)
            - Beta-Binomial: fitted parameters are α (expected number of claims) and β (expected number of "non-claims")
    """
    # Fit the Uniform distribution to the training data
    _, n_fitted_uniform, _, uniform_pmf, neg_log_likelihood_uniform = fit_uniform(train_data)
    test_log_likelihood_uni = -neg_log_likelihood_uniform([n_fitted_uniform], test_data)
    bic_uni = calc_BIC(test_log_likelihood_uni, 1, len(train_data))

    # Fit the Poisson distribution to the training data
    _, λ, _, neg_log_likelihood_poi = fit_poi(train_data)
    test_log_likelihood_poi = -neg_log_likelihood_poi(λ, test_data)
    bic_poi = calc_BIC(test_log_likelihood_poi, 1, len(train_data))

    # Fit the Binomial distribution to the training data
    _, n_fitted_binom, p_fitted_binom, _, _, _, _ = fit_binom(train_data)
    test_log_likelihood_binom = -neg_log_likelihood_binom((np.array([n_fitted_binom]),np.array([p_fitted_binom])), test_data)
    bic_binom = calc_BIC(test_log_likelihood_binom, 2, len(train_data))

    # Fit the Negative Binomial distribution to the training data
    _, r_nb_fitted, p_nb_fitted, _, _, _, _ = fit_nb(train_data, 1) # Run single core to avoid conflicts
    test_log_likelihood_nb = -neg_log_likelihood_nb((r_nb_fitted, p_nb_fitted), test_data)
    bic_nb = calc_BIC(test_log_likelihood_nb, 2, len(train_data))

    # Fit the Beta-Binomial distribution to the training data
    _, alpha_bb_fitted, beta_bb_fitted, _, beta_binomial_pmf, neg_log_likelihood_bb = fit_beta_binom(train_data)
    test_log_likelihood_bb = -neg_log_likelihood_bb((alpha_bb_fitted, beta_bb_fitted), test_data)
    bic_bb = calc_BIC(test_log_likelihood_bb, 2, len(train_data))

    # Won't bother fitting the Empirical distribution since it won't factor into the blended model
    
    # Define the model names
    model_names = ['Uniform', 'Poisson', 'Binomial', 'Negative Binomial', 'Beta-Binomial']

    # Define the number of parameters for each model
    param_counts = [1, 1, 2, 2, 2]

    # Define the BIC values for each model
    bic_values = np.array([bic_uni, bic_poi, bic_binom, bic_nb, bic_bb])

    # Define the PMF values for each model
    pmf_values = [uniform_pmf(x, n_fitted_uniform),
                    poisson.pmf(x, λ),
                    binom.pmf(x, n_fitted_binom, p_fitted_binom),
                    nbinom.pmf(x, r_nb_fitted, p_nb_fitted),
                    beta_binomial_pmf(x, max(train_data), alpha_bb_fitted, beta_bb_fitted)]
    
    params_used = [[n_fitted_uniform],
                   [λ],
                   [n_fitted_binom, p_fitted_binom],
                   [r_nb_fitted, p_nb_fitted],
                   [alpha_bb_fitted, beta_bb_fitted]]

    # Perform Bayesian Model Averaging
    weights, average_pmf, log_likelihood_bma, num_params_bma, bic_averaged_model = bma_using_bic(model_names,
                                                                                                    param_counts,
                                                                                                    bic_values,
                                                                                                    pmf_values,
                                                                                                    train_data,
                                                                                                    test_data)

    return weights, average_pmf, log_likelihood_bma, num_params_bma, bic_averaged_model, params_used


def cross_validate(model_func, data, x, k=5, random_state=42):
    """
    Perform k-fold cross-validation on the given model function and data.
    
    Parameters:
    - model_func: Function to train and evaluate the model. Should return a performance metric.
    - data: The dataset to perform cross-validation on.
    - k: Number of folds for cross-validation.
    
    Returns:
    - mean_performance: Mean performance metric across all folds.
        weights_bma_cv: Weights of the component models within the BMA model.
        average_pmf_bma_cv: PMF of the BMA model.
        log_likelihood_bma_cv: Log-likelihood of the BMA model.
        param_num_bma_cv: The number of parameters of the BMA model
        bic_bma_cv: The Bayesian Information Criterion (BIC) of the averaged model.
        model_uncertainty: The uncertainty of the individual base models.
        params_used: The parameters used in each model.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    model_names = []
    param_counts = []
    bic_values = []
    pmf_values = []

    model_uncertainty = [] # Track uncertainty of the individual models
    params_used_list = [] # Track the parameters used in each model

    for fold_number, (train_index, test_index) in enumerate(kf.split(data), 1):
        logging.info(f"## Evaluating Fold ~{fold_number}~ ##")
        train_data = [data[i] for i in train_index]
        test_data = [data[i] for i in test_index]
        weights, average_pmf, log_likelihood_bma, params_averaged_model, bic_averaged_model, params_used = model_func(train_data, test_data, x)

        # Store the results of BMA for each fold
        model_names.append(f"BMA Model from Fold {fold_number}")
        param_counts.append(params_averaged_model)
        bic_values.append(bic_averaged_model)
        pmf_values.append(average_pmf)
        model_uncertainty.append(weights)
        params_used_list.append(params_used)

    # Perform BMA of the fold models
    weights_bma_cv, average_pmf_bma_cv, log_likelihood_bma_cv, param_num_bma_cv, bic_bma_cv = bma_using_bic(model_names,
                                                                                                            param_counts,
                                                                                                            bic_values,
                                                                                                            pmf_values,
                                                                                                            train_data,
                                                                                                            test_data)
    return weights_bma_cv, average_pmf_bma_cv, log_likelihood_bma_cv, param_num_bma_cv, bic_bma_cv, model_uncertainty, params_used_list

def bootstrap_sample_evaluation(model_func, data, x, random_state, k_folds=5):
    """
    Evaluate a single bootstrap sample.
    
    Parameters:
    - model_func: Function to train and evaluate the model. Should return a performance metric.
    - data: The dataset to perform bootstrapping on.
    - x: Values for the PMF.
    - random_state: Random state for reproducibility.
    - k_folds: Number of folds for cross-validation.
    
    Returns:
    - result: Dictionary containing the evaluation results.
        weights_bma_cv: Weights of the component models within the BMA model.
        average_pmf_bma_cv: PMF of the BMA model.
        log_likelihood_bma_cv: Log-likelihood of the BMA model.
        param_num_bma_cv: The number of parameters of the BMA model
        bic_bma_cv: The Bayesian Information Criterion (BIC) of the averaged model.
        model_uncertainty: The uncertainty of the individual base models.
        params_used: The parameters used in each model.
    """
    # Generate a bootstrap sample
    try:
        logging.info(f"# Generating bootstrap sample with random state ~{random_state}~ #")
        # Assess memory
        logging.info(f"  Memory usage: {psutil.virtual_memory().percent}%")
        # Assess CPU usage
        logging.info(f"  CPU usage: {psutil.cpu_percent(interval=1)}%")
        
        # Generate a bootstrap sample
        bootstrap_sample = resample(data, replace=True, n_samples=len(data), random_state=random_state)
        
        # Perform cross-validation on the bootstrap sample
        k_fold_random_state = random_state*k_folds+len(bootstrap_sample) # Use a different random state for each fold
        weights_bma_cv, average_pmf_bma_cv, log_likelihood_bma_cv, param_num_bma_cv, bic_bma_cv, model_uncertainty, params_used = cross_validate(model_func, bootstrap_sample, x, k=k_folds, random_state=k_fold_random_state)
        
        # Store the results
        result = {
            'random_state': random_state,
            'weights_bma_cv': weights_bma_cv,
            'average_pmf_bma_cv': average_pmf_bma_cv,
            'log_likelihood_bma_cv': log_likelihood_bma_cv,
            'param_num_bma_cv': param_num_bma_cv,
            'bic_bma_cv': bic_bma_cv,
            'model_uncertainty': model_uncertainty,
            'params_used': params_used
        }
        
        return result
    except Exception as e:
        logging.info(f"Error in bootstrap_sample_evaluation: {e}")
        return None

def bootstrap_uncertainty(model_func, data, x, n_bootstraps=1000, k_folds=5, max_workers=None, timeout=None, outfile=None):
    """
    Perform bootstrapping to evaluate parameter uncertainty.
    
    Parameters:
    - model_func: Function to train and evaluate the model. Should return a performance metric.
    - data: The dataset to perform bootstrapping on.
    - x: Values for the PMF.
    - n_bootstraps: Number of bootstrap samples.
    - k_folds: Number of folds for cross-validation.
    - max_workers: Maximum number of workers for parallel processing.
    - timeout: Timeout for each bootstrap sample.
    
    Returns:
    - bootstrap_results: List of results dictionaries from each bootstrap sample.
        weights_bma_cv: Weights of the component (Cross-Validation) models within the Bagged model.
        average_pmf_bma_cv: Probability Mass Function (PMF) of the Bagged model.
            - The mass values for each discrete point on the x-axis, starting from 0.
        log_likelihood_bma_cv: Log-likelihood of the Bagged model.
        param_num_bma_cv: The number of parameters of the Bagged model
        bic_bma_cv: The Bayesian Information Criterion (BIC) of the averaged model.
        model_uncertainty: The uncertainty of the individual model.
            - The weights given to each base model within each Cross-Validation fold model.
        params_used: The parameters used in each model.
            Description of params_used in bootstrap results:
            - Each bootstrap resample comes with the list of fitted parameters used in all of its cross-validation folds
            - At the core, params_used is a list of nested lists containing the fitted parameters in an organized way
            - Each top-line list in params_used corresponds to a cross-validation fold
            - Each fold, in turn, contains a list of the fitted base models. At the time of this writing, there were 5 base models:
                - Uniform: fitted parameter is n (maximal value)
                - Poisson: fitted parameter is λ (mean)
                - Binomial: fitted parameters are n (number of trials) and p (probability)
                - Negative Binomial: fitted parameters are r (number of claims until stop) and p (probability of a claim)
                - Beta-Binomial: fitted parameters are α (expected number of claims) and β (expected number of "non-claims")
    """
    logging.info(f"Starting bootstrapping with {n_bootstraps} samples for {k_folds} cv folds on {max_workers} workers.")
    bootstrap_results = []
    start_time = time.time() # track runtime

    # If max_workers is not specified, use one less than the number of CPU cores
    if max_workers is None:
        # Get the number of CPU cores and set max_workers to be one less
        num_cores = os.cpu_count()
        max_workers = num_cores if num_cores > 1 else 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(bootstrap_sample_evaluation, model_func, data, x, i, k_folds) for i in range(n_bootstraps)]
        
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result(timeout=timeout)
                logging.info(f"Completed Bootstrap Sample {i+1} weights: {result['weights_bma_cv']}")
                if result is not None:
                    bootstrap_results.append(result)
                    if outfile is not None:
                        with open(outfile, 'a') as file:
                            file.write(json.dumps(make_serializable(result), indent=2) + ',\n')
            except Exception as e:
                logging.info(f"Error in future {i}: {e}")
            
            # Calculate elapsed time and estimate remaining time
            elapsed_time = time.time() - start_time
            remaining_time = (elapsed_time / (i + 1)) * (n_bootstraps - (i + 1))
            logging.info(f"# Bootstrap Sample {i+1}/{n_bootstraps} completed. Estimated time remaining: {remaining_time:.2f} seconds")

    return bootstrap_results


# Configure logging to save to a file
logging.basicConfig(filename='output.log', level=logging.INFO)
