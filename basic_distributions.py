import logging
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import psutil
from scipy.optimize import minimize
from scipy.special import betaln, comb
from scipy.stats import binom, nbinom, poisson


def fit_uniform(y):
    """
    Fit a Uniform distribution to the data y using maximum likelihood estimation.

    Parameters
    ----------
    y : array_like of shape (n_samples,): Data to fit.

    Returns
    -------
    log_likelihood_uniform : float
        Log-likelihood of the Uniform model.
    n_fitted : float
        Fitted parameter for the Uniform distribution.
    bic_uniform : float
        Bayesian Information Criterion (BIC) of the Uniform model.
    uniform_pmf : function
        Probability mass function of the Uniform distribution.
    neg_log_likelihood_uniform : function
        Negative log-likelihood function of the Uniform distribution.
    """

    # Define the probability mass function (pmf) of the Uniform distribution
    def uniform_pmf(k, n):
        k = np.asarray(k)  # Ensure k is a numpy array
        return [1 / n if j <= n else 0 for j in k]

    # Define the negative log-likelihood function for the Uniform distribution
    def neg_log_likelihood_uniform(params, data):
        n = params[0]  # Assuming n is the number of trials in the data
        pmf_values = uniform_pmf(data, n)
        # Avoid log(0) by replacing 0 with a very small number
        pmf_values = np.where(np.array(pmf_values) == 0, 1e-10, pmf_values)
        return -np.sum(np.log(pmf_values))

    n_range = range(1,np.max(y)*10)

    # Grid search for the best n
    best_n = None
    best_neg_log_likelihood = np.inf

    for n in n_range:
        # Initial guess for n
        initial_guess = np.array([n])
        # Minimize the negative log-likelihood
        result = minimize(neg_log_likelihood_uniform, initial_guess, args=(y,), bounds=[(1e-5, None)])

        if result.fun < best_neg_log_likelihood:
            best_neg_log_likelihood = result.fun
            best_n = result.x[0]

    # Extract the fitted parameter
    n_fitted = best_n

    # Calculate the log-likelihood of the model
    log_likelihood_uniform = -neg_log_likelihood_uniform([n_fitted], y)

    # Number of parameters (k) and number of data points (n)
    k = 1
    n = len(y)

    # Calculate BIC
    bic_uniform = k * np.log(n) - 2 * log_likelihood_uniform

    return log_likelihood_uniform, n_fitted, bic_uniform, uniform_pmf, neg_log_likelihood_uniform


def fit_poi(y):
    """
    Fit a Poisson distribution to the data y using maximum likelihood estimation.

    Parameters
    ----------
    y : array_like of shape (n_samples,): Data to fit.

    Returns
    -------
    log_likelihood_poi : float
        Log-likelihood of the Poisson model.
    λ : float
        Fitted parameter for the Poisson distribution.
    bic_poi : float
        Bayesian Information Criterion (BIC) of the Poisson model.
    neg_log_likelihood_poi : function
        Negative log-likelihood function of the Poisson distribution
    """
    λ = np.mean(y)  # MLE for λ in Poisson distribution

    #Define the negative log-likelihood function for the Poisson distribution
    def neg_log_likelihood_poi(λ, data):
        return -np.sum(poisson.logpmf(data, λ))

    # Calculate the log-likelihood of the Poisson model
    log_likelihood_poi = -neg_log_likelihood_poi(λ, y)

    # Number of parameters (k) and number of data points (n)
    k = 1
    n = len(y)

    # Calculate the Bayesian Information Criterion (BIC)
    bic_poi = k * np.log(n) - 2 * log_likelihood_poi

    return log_likelihood_poi, λ, bic_poi, neg_log_likelihood_poi

# Define the negative log-likelihood function for the Binomial model
def neg_log_likelihood_binom(params, y):
    n, p = params # Unpack the parameters
    pmf_values = binom.pmf(y, n, p)
    # Avoid log(0) by replacing 0 with a very small number
    pmf_values = np.where(np.array(pmf_values) == 0, 1e-10, pmf_values)
    result = -np.sum(np.log(pmf_values))
    return result

def fit_binom(y):
    """
    Fit a Binomial distribution to the data y using maximum likelihood estimation.

    Parameters
    ----------
    y : array_like of shape (n_samples,): Data to fit.

    Returns
    -------
    log_likelihood_binom : float
        Log-likelihood of the Binomial model.
    n_fitted_binom : float
        Fitted parameter for the Binomial distribution.
    p_fitted_binom : float
        Fitted parameter for the Binomial distribution.
    bic_binom : float
        Bayesian Information Criterion (BIC) of the Binomial model.
    n_grid : array_like
        Grid of fitted n values.
    p_grid : array_like
        Grid of fitted p values.
    log_likelihood_grid : array_like
        Grid of fitted log-likelihood values.
    neg_log_likelihood_binom : function
        Negative log-likelihood function of the Binomial distribution
    """

    # Define the grid of parameter ranges
    n_trials_range = np.arange(0, np.max(y)*200 + 1, 5)
    p_range = np.linspace(1e-5, 1 - 1e-5, 100)

    # Vectorized grid search for the best n and p
    n_grid, p_grid = np.meshgrid(n_trials_range, p_range, indexing='ij')
    log_likelihood_grid = np.zeros_like(n_grid, dtype=float)

    for i in range(n_grid.shape[0]):
        for j in range(n_grid.shape[1]):
            n = n_grid[i, j]
            p = p_grid[i, j]
            log_likelihood_grid[i, j] = -neg_log_likelihood_binom((np.array([n]),np.array([p])), np.array(y))
            pass

    # Mask out np.inf values by setting them to a very large number
    log_likelihood_grid[np.isinf(log_likelihood_grid)] = 1e10

    # Find the indices of the minimum negative log-likelihood
    min_idx = np.unravel_index(np.argmax(log_likelihood_grid), log_likelihood_grid.shape)
    n_fitted_binom = n_grid[min_idx]
    p_fitted_binom = p_grid[min_idx]
    best_neg_log_likelihood = log_likelihood_grid[min_idx]

    # Calculate the log-likelihood
    log_likelihood_binom = best_neg_log_likelihood

    # Number of parameters (k)
    k = 2

    # Number of data points (n)
    n = len(y)

    # Calculate BIC
    bic_binom = k * np.log(n) - 2 * log_likelihood_binom

    return log_likelihood_binom, n_fitted_binom, p_fitted_binom, bic_binom, n_grid, p_grid, log_likelihood_grid
    # Define the negative log-likelihood function for the Negative Binomial distribution

def neg_log_likelihood_nb(params, y):
    r, p = params # Unpack the parameters
    pmf_values = nbinom.pmf(y, r, p)
    # Avoid log(0) by replacing 0 with a very small number
    pmf_values = np.where(np.array(pmf_values) == 0, 1e-10, pmf_values)
    result = -np.sum(np.log(pmf_values))
    return result

# Function to perform optimization for a given r
def optimize_for_r(i, r, y, p_range):
    best_p = None
    best_neg_log_likelihood = np.inf
    local_log_likelihood_matrix = np.zeros(len(p_range))
    for j, p in enumerate(p_range):
        initial_guess = np.array([r, p])
        result = neg_log_likelihood_nb(initial_guess, y)
        # result = minimize(neg_log_likelihood_nb, initial_guess, args=(y,), bounds=[(1e-5, None), (1e-5, 1-1e-5)])

        # Store the results in the local matrix
        local_log_likelihood_matrix[j] = result

        if result < best_neg_log_likelihood:
            best_neg_log_likelihood = result
            best_p = p
    return i, local_log_likelihood_matrix, r, best_p, best_neg_log_likelihood

def fit_nb(y, max_workers=None):
    """
    Fit a Negative Binomial distribution to the data y using maximum likelihood estimation.

    Parameters
    ----------
    y : array_like of shape (n_samples,): Data to fit.

    Returns
    -------
    log_likelihood_nb : float
        Log-likelihood of the Negative Binomial model.
    r_nb_fitted : float
        Fitted parameter for the Negative Binomial distribution.
    p_nb_fitted : float
        Fitted parameter for the Negative Binomial distribution.
    bic_nb : float
        Bayesian Information Criterion (BIC) of the Negative Binomial model.
    r_range : array_like
        Grid of fitted r values.
    p_range : array_like
        Grid of fitted p values.
    log_likelihood_matrix : array_like
        Grid of fitted log-likelihood values.
    neg_log_likelihood_nb : function
        Negative log-likelihood function of the Negative Binomial distribution
    """
    # Define the grid of parameter ranges
    r_range = np.linspace(1, np.max(y) + 1, 100)  # Adjusted range for r
    p_range = np.linspace(1e-5, 1 - 1e-5, 1000)  # Adjusted range for p

    # Initialize an empty matrix to store the results
    log_likelihood_matrix = np.zeros((len(r_range), len(p_range)))

    # If no max_workers provides, determine the number of workers based on CPU utilization
    if max_workers is None:
        # Get the utilization of each core
        core_utilizations = psutil.cpu_percent(percpu=True)

        # Count the number of cores with utilization under 75%
        underutilized_cores = sum(1 for utilization in core_utilizations if utilization < 75)

        # Set n_jobs to the minimum of the number of underutilized cores and a specified maximum (e.g., 4)
        max_jobs = os.cpu_count() - 1
        max_workers = min(underutilized_cores, max_jobs)

    results = []

    if max_workers == 1: # Don't parallelize if only 1 worker
        for i, r in enumerate(r_range):
            results.append(optimize_for_r(i, r, y, p_range))
    else: # Parallel processing to optimize for each r
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(optimize_for_r, i, r, y, p_range) for i, r in enumerate(r_range)]
            
            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result(timeout=60)
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logging.info(f"Error in Neg Bin fit {i}: {e}")

    # Update the global log_likelihood_matrix with results from parallel processing
    for result in results:
        i, local_log_likelihood_matrix, r, best_p, best_neg_log_likelihood = result
        log_likelihood_matrix[i, :] = local_log_likelihood_matrix

    # Find the best parameters from the results
    best_result = min(results, key=lambda x: x[4])
    best_r, best_p, best_neg_log_likelihood = best_result[2], best_result[3], best_result[4]

    r_nb_fitted = best_r
    p_nb_fitted = best_p

    # Calculate the log-likelihood
    log_likelihood_nb = best_neg_log_likelihood

    # Number of parameters (k) and number of data points (n)
    k = 2
    n = len(y)

    # Calculate BIC
    bic_nb = k * np.log(n) + 2 * log_likelihood_nb

    return log_likelihood_nb, r_nb_fitted, p_nb_fitted, bic_nb, r_range, p_range, log_likelihood_matrix

def fit_beta_binom(y):
    """
    Fit a Beta-Binomial distribution to the data y using maximum likelihood estimation.

    Parameters
    ----------
    y : array_like of shape (n_samples,): Data to fit.

    Returns
    -------
    log_likelihood_bb : float
        Log-likelihood of the Beta-Binomial model.
    alpha_bb_fitted : float
        Fitted parameter for the Beta-Binomial distribution.
    beta_bb_fitted : float
        Fitted parameter for the Beta-Binomial distribution.
    bic_bb : float
        Bayesian Information Criterion (BIC) of the Beta-Binomial model.
    beta_binomial_pmf : function
        Probability mass function of the Beta-Binomial distribution.
    neg_log_likelihood_bb : function
        Negative log-likelihood function of the Beta-Binomial
    """
    # Define the Beta-Binomial PMF
    def beta_binomial_pmf(k, n, alpha, beta):
        k = np.asarray(k)  # Ensure k is a numpy array
        binom_coeff = comb(n, k)  # Calculate the binomial coefficient
        pmf = binom_coeff * np.exp(betaln(k + alpha, n - k + beta) - betaln(alpha, beta))
        return pmf

    # Define the negative log-likelihood function for the Beta-Binomial distribution
    def neg_log_likelihood_bb(params, data):
        alpha, beta = params
        n = max(data)  # Assuming n is the maximal value in the data
        return -np.sum(np.log(beta_binomial_pmf(data, n, alpha, beta)))

    # Initial guesses for alpha and beta
    initial_params = [1, 1]

    # Fit the Beta-Binomial distribution to the data
    model_bb = minimize(neg_log_likelihood_bb, initial_params, args=(y,), bounds=[(1e-5, None), (1e-5, None)])

    # Extract the fitted parameters
    alpha_bb_fitted, beta_bb_fitted = model_bb.x

    # Calculate the log-likelihood of the model
    log_likelihood_bb = -neg_log_likelihood_bb((alpha_bb_fitted, beta_bb_fitted), y)

    # Number of parameters (k) and number of data points (n)
    k = 2
    n = len(y)

    # Calculate BIC
    bic_bb = k * np.log(n) - 2 * log_likelihood_bb

    return log_likelihood_bb, alpha_bb_fitted, beta_bb_fitted, bic_bb, beta_binomial_pmf, neg_log_likelihood_bb


def fit_empirical(y):
    """
    Fit an Empirical distribution to the data y using maximum likelihood estimation.

    Parameters
    ----------
    y : array_like of shape (n_samples,): Data to fit.

    Returns
    -------
    log_likelihood : float
        Log-likelihood of the Empirical model.
    probabilities : dict
        Dictionary containing the probabilities of each unique value in the data.
    bic_emp : float
        Bayesian Information Criterion (BIC) of the Empirical model.
    predictive_distribution : function
        Predictive distribution of the Empirical distribution.
    neg_log_likelihood_emp : function
        Negative log-likelihood function of the Empirical distribution.
    """
    # Calculate the frequency of each value in the observed data
    frequency_counts = Counter(y)

    # Normalize the frequencies to get probabilities
    total_count = sum(frequency_counts.values())
    probabilities = {k: v / total_count for k, v in frequency_counts.items()}

    # Create the predictive distribution
    def predictive_distribution(value):
        return probabilities.get(value, 0)
    
    # Define the negative log-likelihood function for the Empirical distribution
    def neg_log_likelihood_emp(data):
        return -np.sum([np.log(predictive_distribution(val)) for val in data])

    # Calculate the log-likelihood of the observed data
    log_likelihood = -neg_log_likelihood_emp(y)

    # Number of parameters (k) and number of data points (n)
    k = len(y)
    n = len(y)

    # Calculate BIC
    bic_emp = k * np.log(n) - 2 * log_likelihood

    return log_likelihood, probabilities, bic_emp, predictive_distribution, neg_log_likelihood_emp


# Configure logging to save to a file
logging.basicConfig(filename='output.log', level=logging.INFO)
