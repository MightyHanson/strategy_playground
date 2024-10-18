# optimizer.py

from cvxpy import Variable, Problem, Minimize
import cvxpy as cp
import pandas as pd
import numpy as np
import logging

def optimize_portfolio(expected_returns, cov_matrix, risk_aversion=0.5):
    """
    Optimize the portfolio based on expected returns and covariance matrix using Mean-Variance Optimization.

    Args:
        expected_returns (pd.Series): Expected returns for each asset.
        cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
        risk_aversion (float): Risk aversion parameter (higher means more risk-averse).

    Returns:
        pd.Series: Optimized weights for each asset.
    """
    n = len(expected_returns)
    weights = cp.Variable(n)

    # Regularize the covariance matrix
    epsilon = 1e-4
    cov_matrix += np.eye(cov_matrix.shape[0]) * epsilon

    # Wrap the covariance matrix with psd_wrap
    P = cp.psd_wrap(cov_matrix.values)

    # Define the objective: Minimize the risk (variance) minus expected return scaled by risk aversion
    objective = cp.Minimize(risk_aversion * cp.quad_form(weights, P) - expected_returns.values @ weights)

    # Constraints: weights sum to 1, weights >= 0 (long-only)
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0
    ]

    prob = Problem(objective, constraints)

    try:
        prob.solve(solver=cp.OSQP, verbose=True)
    except Exception as e:
        logging.error(f"OSQP solver failed: {e}")
        raise

    if weights.value is None:
        logging.error("Optimization failed to find a solution.")
        raise ValueError("Optimization failed to find a solution.")

    # Convert weights to a pandas Series
    optimized_weights = pd.Series(weights.value, index=expected_returns.index)
    optimized_weights = optimized_weights.clip(lower=0)  # Ensure no negative weights
    optimized_weights /= optimized_weights.sum()  # Normalize to sum to 1

    return optimized_weights
