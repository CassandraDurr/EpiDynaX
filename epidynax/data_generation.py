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
        list: List containing the derivatives [dS, dI, dR] at each time point.
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


def ode_solver(par: list, time: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """Solve the SIR model using the odeint function from scipy.integrate.

    Args:
        par (list): List containing the initial values of S, I, and R.
        time (np.ndarray): Array of time points at which to solve the ODE.
        beta (float): Infection rate of the disease.
        gamma (float): Recovery rate of the disease.

    Returns:
        np.ndarray: Array containing the solution of the SIR model at each time point.
    """
    # Solve the SIR model using odeint
    result = odeint(sir_model, par, time, args=(beta, gamma))
    return result
