"""Module for ODEFormer utility functions."""

import io
from contextlib import redirect_stdout

import numpy as np
from matplotlib import pyplot as plt
from odeformer.model import SymbolicTransformerRegressor


def get_odeformer_model(
    beam_size: int = 50, beam_temperature: float = 0.1, rescale: bool = True
) -> SymbolicTransformerRegressor:
    """Return a pre-trained ODEFormer model with specified beam parameters.

    Args:
        beam_size (int, optional): Beam size. Defaults to 50.
        beam_temperature (float, optional): Beam temperature. Defaults to 0.1.
        rescale (bool, optional): Whether to rescale. Defaults to True.

    Returns:
        SymbolicTransformerRegressor: Pre-trained ODEFormer model.
    """
    # Initialise the SymbolicTransformerRegressor with pre-trained weights
    dstr = SymbolicTransformerRegressor(from_pretrained=True, rescale=rescale)
    dstr.set_model_args({"beam_size": beam_size, "beam_temperature": beam_temperature})

    return dstr


def plot_actual_vs_estimated_trajectory(
    times: np.ndarray,
    trajectory: np.ndarray,
    pred_trajectory: np.ndarray,
    figsize: tuple[int, int] = (12, 8),
    ylabel: str = "Population",
) -> plt.Figure:
    """Plot the actual vs estimated trajectory of an SIR model.

    Args:
        times (np.ndarray): The time scale/ x axis.
        trajectory (np.ndarray): The true model trajectory.
        pred_trajectory (np.ndarray): The estimated trajectory of the SIR model.
        figsize (tuple[int, int], optional): The plot dimension. Defaults to (12, 8).
        ylabel (str, optional): The y-axis label. Defaults to "Population".

    Returns:
        plt.Figure: The actual vs estimated trajectory plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    # Actuals
    ax.scatter(
        times,
        trajectory[:, 0],
        label="Susceptible (Actual)",
        color="blue",
        marker="o",
        alpha=0.3,
    )
    ax.scatter(
        times,
        trajectory[:, 1],
        label="Infected (Actual)",
        color="red",
        marker="o",
        alpha=0.3,
    )
    ax.scatter(
        times,
        trajectory[:, 2],
        label="Recovered (Actual)",
        color="green",
        marker="o",
        alpha=0.3,
    )
    # Estimates
    ax.plot(
        times,
        pred_trajectory[:, 0],
        label="Susceptible (Estimated)",
        color="blue",
        linestyle="dashed",
    )
    ax.plot(
        times,
        pred_trajectory[:, 1],
        label="Infected (Estimated)",
        color="red",
        linestyle="dashed",
    )
    ax.plot(
        times,
        pred_trajectory[:, 2],
        label="Recovered (Estimated)",
        color="green",
        linestyle="dashed",
    )
    # Plot settings
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(ylabel)
    ax.set_title("Actual vs Estimated SIR Model Trajectory")
    ax.legend()
    fig.tight_layout()

    return fig


# pylint: disable=invalid-name
def print_SIR_equations(dstr: SymbolicTransformerRegressor) -> None:
    """Print the SIR model equations from the ODEFormer output."""
    f = io.StringIO()
    with redirect_stdout(f):
        dstr.print(n_predictions=1)
    output = f.getvalue()
    output = output.replace("x_0", "S").replace("x_1", "I").replace("x_2", "R")
    print(output)
