"""Module which generates S, I, R data using odeint."""

import numpy as np
from scipy.integrate import odeint


# pylint: disable=unused-argument
def sir_model(par: list, time: np.ndarray, beta: float, gamma: float) -> list:
    """SIR model differential equations.

    Args:
        par (list): List containing the current values of S, I, and R.
        time (np.ndarray): Array of time points at which to solve the ODE.
        beta (float): Infection rate of the disease.
        gamma (float): Recovery rate of the disease.

    Raises:
        ValueError: Require total population to be greater than 0.
        ValueError: Susceptible, infected, and recovered counts must be non-negative.
        ValueError: Beta and gamma must be non-negative.

    Returns:
        list: List containing the derivatives at each time point.
    """
    susceptible, infected, recovered = par
    population_size = susceptible + infected + recovered
    # Error handling for input parameters
    if population_size <= 0:
        raise ValueError("Total population N must be greater than 0")
    if susceptible < 0 or infected < 0 or recovered < 0:
        raise ValueError(
            "Susceptible, infected, and recovered counts must be non-negative"
        )
    if beta < 0 or gamma < 0:
        raise ValueError("Beta and gamma must be non-negative")
    # SIR model differential equations
    derivative_susceptible = -beta * susceptible * infected / population_size
    derivative_infected = (
        beta * susceptible * infected / population_size - gamma * infected
    )
    derivative_recovered = gamma * infected
    # Return the derivatives as a list
    return [derivative_susceptible, derivative_infected, derivative_recovered]


# pylint: disable=unused-argument
def sir_model_proportion(
    par: list, time: np.ndarray, beta: float, gamma: float
) -> list:
    """SIR model using proportions (S, I, R as fractions of total population).

    Args:
        par (list): List containing current values of S, I, and R (proportions).
        time (np.ndarray): Array of time points.
        beta (float): Infection rate.
        gamma (float): Recovery rate.

    Raises:
        ValueError: Proportions must be between 0 and 1 and sum to 1 (within tolerance).
        ValueError: Beta and gamma must be non-negative.

    Returns:
        list: Derivatives at each time point.
    """
    susceptible, infected, recovered = par
    total = susceptible + infected + recovered
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError("S, I, R proportions must sum to 1")
    if any(x < 0 or x > 1 for x in [susceptible, infected, recovered]):
        raise ValueError("S, I, R proportions must be between 0 and 1")
    if beta < 0 or gamma < 0:
        raise ValueError("Beta and gamma must be non-negative")
    derivative_susceptible = -beta * susceptible * infected
    derivative_infected = beta * susceptible * infected - gamma * infected
    derivative_recovered = gamma * infected
    return [derivative_susceptible, derivative_infected, derivative_recovered]


def ode_solver(
    par: list, time: np.ndarray, beta: float, gamma: float, proportion: bool = True
) -> np.ndarray:
    """Solve the SIR model using the odeint function from scipy.integrate.

    Args:
        par (list): List containing the initial values of S, I, and R.
        time (np.ndarray): Array of time points at which to solve the ODE.
        beta (float): Infection rate of the disease.
        gamma (float): Recovery rate of the disease.
        proportion (bool): If True, use proportions for S, I, R; otherwise use counts.

    Returns:
        np.ndarray: Array containing the solution of the SIR model at each time point.
    """
    # Solve the SIR model using odeint
    if proportion:
        result = odeint(sir_model_proportion, par, time, args=(beta, gamma))
    else:
        result = odeint(sir_model, par, time, args=(beta, gamma))
    return result
