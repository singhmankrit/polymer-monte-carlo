import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import scipy.optimize as opt


def plot(
    lengths: NDArray[np.int64],
    r2: NDArray[np.float64],
    weights: NDArray[np.longdouble],
    dim: int,
    alive: NDArray[np.bool],
    max_step: int,
    axis_name: str,
    label: str,
    title: str,
    file_name: str,
):
    """
    Creates a plot for a length-dependent observable with a twinx that contains
    the number of polymers of at least that length

    Parameters
        lengths (ndarray): array of the lengths the points are given at, the values for the X-axis
        r2 (ndarray): array containing the values of the respective observable at each length
        weights (ndarray): array containing the weights of each of the observables, for calculating error and mean
        dim (int): the dimension of the simulation, for which fit to use
        alive (ndarray): array containing a mask of when the polymers still exist
        max_step (int): the size of the longest polymer in the dataset
        axis_name (str): the label to put on the left axis
        label (str): the name for the label in the legend
        title (str): the title for the plot
        file_name (str): what file to save the plot to
    """
    mean_r2, error_r2 = analytical_error(r2, weights, alive)

    fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_xlabel(r"L ($\sigma$)")
    ax.set_ylabel(axis_name)

    ax.plot(lengths, mean_r2, label=label, color="C0")
    ax.fill_between(
        lengths,
        mean_r2 - error_r2,
        mean_r2 + error_r2,
        alpha=0.3,
        color="C0",
        label="error",
    )
    # Fit model depending on dimension
    if dim == 2:
        opt_params, _ = opt.curve_fit(growth_model, lengths, mean_r2)
        ax.plot(
            lengths,
            opt_params[0] * lengths ** (3 / 2),
            label=f"best fit: ${opt_params[0]:.03f} L^{{3/2}}$",
            color="C1",
        )
    elif dim == 3:
        opt_params, _ = opt.curve_fit(growth_model_3, lengths, mean_r2)
        ax.plot(
            lengths,
            opt_params[0] * lengths ** (6 / 5),
            label=f"best fit: ${opt_params[0]:.03f} L^{{6/5}}$",
            color="C1",
        )

    # Show number of polymers on secondary y-axis
    ax_right = ax.twinx()
    ax_right.set_ylabel("number of polymers")
    ax_right.set_yscale("log")
    ax_right.plot(
        lengths,
        np.sum(alive[:, :max_step], axis=0),
        alpha=0.5,
        label="number of polymers",
        color="C2",
    )

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    print(lines2, labels2)
    ax.legend(lines + lines2, labels + labels2, loc=0)

    plt.tight_layout()
    fig.savefig(file_name)
    plt.close()


def plot_gyration(
    lengths: NDArray[np.int64],
    r2: NDArray[np.float64],
    weights: NDArray[np.longdouble],
    dim: int,
    alive: NDArray[np.bool],
    max_step: int,
):
    """
    Creates a plot for the gyration with a twinx that contains
    the number of polymers of at least that length

    Parameters
        lengths (ndarray): array of the lengths the points are given at, the values for the X-axis
        r2 (ndarray): array containing the values of the gyration at each length
        weights (ndarray): array containing the weights of each of the observables, for calculating error and mean
        dim (int): the dimension of the simulation, for which fit to use
        alive (ndarray): array containing a mask of when the polymers still exist
        max_step (int): the size of the longest polymer in the dataset
    """
    plot(
        lengths,
        r2,
        weights,
        dim,
        alive,
        max_step,
        "Length dependent radius of Gyration",
        r"Radius of Gyration ($\sigma^2$)",
        "gyration",
        "gyration.png",
    )


def plot_end_to_end(
    lengths: NDArray[np.int64],
    r2: NDArray[np.float64],
    weights: NDArray[np.longdouble],
    dim: int,
    alive: NDArray[np.bool],
    max_step: int,
):
    """
    Creates a plot for the end-to-end distance with a twinx that contains
    the number of polymers of at least that length

    Parameters
        lengths (ndarray): array of the lengths the points are given at, the values for the X-axis
        r2 (ndarray): array containing the values of the end-to-end distance at each length
        weights (ndarray): array containing the weights of each of the observables, for calculating error and mean
        dim (int): the dimension of the simulation, for which fit to use
        alive (ndarray): array containing a mask of when the polymers still exist
        max_step (int): the size of the longest polymer in the dataset
    """
    plot(
        lengths,
        r2,
        weights,
        dim,
        alive,
        max_step,
        "Length dependent end-to-end distance",
        r"end to end dist ($\sigma^2$)",
        "end-to-end",
        "end_to_end.png",
    )


def analytical_error(
    r2: NDArray[np.float64], w: NDArray[np.longdouble], alive: NDArray[np.bool]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculates the weighted mean and the analytical error of the observable r2 using the weights in w.

    Parameters
        r2 (ndarray): array containing the values of the observable at each length for each chain
        w (ndarray): array containing the weights for each length of each chain
        alive (ndarray): array containing a boolean mask of whether the chain exists at a length
    """
    _, max_step = r2.shape
    N = np.sum(alive[:, :max_step], axis=0)
    r2_mean = np.sum(w * r2, axis=0) / np.sum(w, axis=0)
    numerator = np.sum((w / np.max(w)) ** 2 * (r2 - r2_mean) ** 2, axis=0)
    denominator = (np.sum(w / np.max(w), axis=0)) ** 2
    error = np.sqrt((N / (N - 1)) * (numerator / denominator))
    return r2_mean, error


def growth_model(L, A):
    """
    a model to fit for the 2D polymer case from [the lecture notes](https://compphys.quantumtinkerer.tudelft.nl/proj2-polymers/#model-polymers-as-a-self-avoiding-random-walk-on-a-lattice)

    Parameters
        A (float): scaling factor for the fit
        L (float): length to estimate at
    """
    return A * L ** (3 / 2)


def growth_model_3(L, A):
    """
    a model to fit for the 3D polymer case from [the lecture notes](https://compphys.quantumtinkerer.tudelft.nl/proj2-polymers/#model-polymers-as-a-self-avoiding-random-walk-on-a-lattice)

    Parameters
        A (float): scaling factor for the fit
        L (float): length to estimate at
    """
    return A * L ** (6 / 5)
