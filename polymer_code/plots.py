import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import scipy.optimize as opt
from matplotlib import animation


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
    observable_mean, observable_error = analytical_error(r2, weights, alive)
    observable_error = np.where(observable_error <= 1e-2, 1e-2, observable_error)
    fig, ax = plt.subplots()
    print(title)
    ax.set_title(title)
    ax.set_xlabel(r"L ($\sigma$)")
    ax.set_ylabel(axis_name)

    ax.plot(lengths, observable_mean, label=label, color="C0")
    ax.fill_between(
        lengths,
        observable_mean - observable_error,
        observable_mean + observable_error,
        alpha=0.3,
        color="C0",
        label="error",
    )

    (model, exp) = (growth_model, 1.5) if dim == 2 else (growth_model_3, 1.2)

    opt_params_fixed, _ = opt.curve_fit(
        model,
        lengths,
        observable_mean.astype(np.float64),
        sigma=observable_error.astype(np.float64),
        absolute_sigma=True,
        nan_policy="omit",
    )
    y_true = observable_mean
    y_pred = opt_params_fixed[0] * lengths**exp
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"R2 score (fixed growth exp): {r2}")
    ax.plot(
        lengths,
        opt_params_fixed[0] * lengths**exp,
        label=f"fixed exp best fit: ${opt_params_fixed[0]:.03f} L^{{{exp}}}$ / R2: {r2:.03f}",
        color="C1",
    )

    opt_params_variable, _ = opt.curve_fit(
        growth_model_custom,
        lengths,
        observable_mean.astype(np.float64),
        sigma=observable_error.astype(np.float64),
        absolute_sigma=True,
    )
    y_true = observable_mean
    y_pred = opt_params_variable[0] * lengths ** (opt_params_variable[1])
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"R2 score (variable growth exp): {r2}")
    ax.plot(
        lengths,
        opt_params_variable[0] * lengths ** (opt_params_variable[1]),
        label=f"variable exp best fit: ${opt_params_variable[0]:.03f} L^{{{opt_params_variable[1]:.2f}}}$ / R2: {r2:.03f}",
        color="C3",
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
    ax.legend(lines + lines2, labels + labels2, loc=0)

    plt.tight_layout()
    fig.savefig(file_name)
    plt.close()


def plot_animation(
    chains: NDArray[np.float64],
    alive: NDArray[np.bool],
    idxs: NDArray[np.int64],
    dimension: int,
):
    """
    Create an animation of the polymer path for the `idxs` polymers

    Parameters
        chains (ndarray): array with all the positions of the chains at every timestep
        alive (ndarray): status of the chains at every length
        idxs (ndarray): array of indeces containing what chains to plot (chains are sorted from long to short)
        dimension (int): the dimension of the chain coordinates
    """

    alive_sum = np.sum(alive, axis=1)
    argsorted = np.argsort(alive_sum)
    sorted = chains[argsorted]
    sorted_len = alive_sum[argsorted]

    fig = plt.figure()
    if dimension == 2:
        ax = fig.add_subplot()
    elif dimension == 3:
        ax = fig.add_subplot(projection="3d")
    title = ax.set_title("selected polymers at step 0")

    def update(num, data, lines, title):
        """
        Helper function for creating the animation

        Parameters
            num (int): what frame the animation is currently
            data (ndarray): array containing the data for the animation
            lines (enumerable(Line)): enumerable of the matplotlib Line objects
            title (Title): matplotlib title object
        """
        title.set_text(f"selected polymers at step {num}")
        for i, line in enumerate(lines):
            if sorted_len[-idxs[i]] > num:
                line.set_data(data[-idxs[i], : num + 1, :2].T)
                if dimension == 3:
                    line.set_3d_properties(data[-idxs[i], : num + 1, 2].T)

    lens = sorted_len[-idxs]

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if dimension == 2:
        ax.set_xlim(
            [np.min(sorted[-idxs, :, 0]) - 0.5, np.max(sorted[-idxs, :, 0]) + 0.5]
        )
        ax.set_ylim(
            [np.min(sorted[-idxs, :, 1]) - 0.5, np.max(sorted[-idxs, :, 1]) + 0.5]
        )
    elif dimension == 3:
        ax.set_zlabel("Z")
        ax.set_xlim3d(
            [np.min(sorted[-idxs, :, 0]) - 0.5, np.max(sorted[-idxs, :, 0]) + 0.5]
        )
        ax.set_ylim3d(
            [np.min(sorted[-idxs, :, 1]) - 0.5, np.max(sorted[-idxs, :, 1]) + 0.5]
        )
        ax.set_zlim3d(
            [np.min(sorted[-idxs, :, 2]) - 0.5, np.max(sorted[-idxs, :, 2]) + 0.5]
        )

    lines = []
    for idx in idxs:
        if dimension == 2:
            (line,) = ax.plot(sorted[-idx, 0:1, 0], sorted[-idx, 0:1, 1])
        elif dimension == 3:
            (line,) = ax.plot(
                sorted[-idx, 0:1, 0], sorted[-idx, 0:1, 1], sorted[-idx, 0:1, 2]
            )
        lines.append(line)

    ani = animation.FuncAnimation(
        fig, update, np.max(lens), fargs=(sorted, lines, title), blit=False
    )
    ani.save("polymers.mp4", writer="ffmpeg", fps=20)


def plot_gyration(
    lengths: NDArray[np.int64],
    gyrations: NDArray[np.float64],
    weights: NDArray[np.longdouble],
    dim: int,
    alive: NDArray[np.bool],
    max_step: int,
    shape: str,
):
    """
    Creates a plot for the gyration with a twinx that contains
    the number of polymers of at least that length

    Parameters
        lengths (ndarray): array of the lengths the points are given at, the values for the X-axis
        gyrations (ndarray): array containing the values of the gyration at each length
        weights (ndarray): array containing the weights of each of the observables, for calculating error and mean
        dim (int): the dimension of the simulation, for which fit to use
        alive (ndarray): array containing a mask of when the polymers still exist
        max_step (int): the size of the longest polymer in the dataset
        shape (str): the shape of the lattice
    """
    plot(
        lengths,
        gyrations,
        weights,
        dim,
        alive,
        max_step,
        "Length dependent radius of Gyration",
        r"Radius of Gyration ($\sigma^2$)",
        "Gyration for " + shape + " lattice " + str(dim) + "D",
        "gyration.png",
    )


def plot_end_to_end(
    lengths: NDArray[np.int64],
    end_to_ends: NDArray[np.float64],
    weights: NDArray[np.longdouble],
    dim: int,
    alive: NDArray[np.bool],
    max_step: int,
    shape: str,
):
    """
    Creates a plot for the end-to-end distance with a twinx that contains
    the number of polymers of at least that length

    Parameters
        lengths (ndarray): array of the lengths the points are given at, the values for the X-axis
        end_to_ends (ndarray): array containing the values of the end-to-end distance at each length
        weights (ndarray): array containing the weights of each of the observables, for calculating error and mean
        dim (int): the dimension of the simulation, for which fit to use
        alive (ndarray): array containing a mask of when the polymers still exist
        max_step (int): the size of the longest polymer in the dataset
        shape (str): the shape of the lattice
    """
    plot(
        lengths,
        end_to_ends,
        weights,
        dim,
        alive,
        max_step,
        "Length dependent end-to-end distance",
        r"end to end dist ($\sigma^2$)",
        "End-To-End distance for " + shape + " lattice " + str(dim) + "D",
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

    Returns
        ndarray: array containing the mean value for each length
        ndarray: array containing the standard deviation for each length
    """
    _, max_step = r2.shape
    N = np.sum(alive[:, :max_step], axis=0)
    r2_mean = np.sum(w * r2, axis=0) / np.sum(w, axis=0)
    numerator = np.sum((w / np.max(w)) ** 2 * (r2 - r2_mean) ** 2, axis=0)
    denominator = (np.sum(w / np.max(w), axis=0)) ** 2
    error = np.sqrt((N / (N - 1 + 1e-4)) * (numerator / denominator))
    return r2_mean, error


def growth_model(L, A):
    """
    a model to fit for the 2D polymer case from [the lecture notes](https://compphys.quantumtinkerer.tudelft.nl/proj2-polymers/#model-polymers-as-a-self-avoiding-random-walk-on-a-lattice)

    Parameters
        A (float): scaling factor for the fit
        L (float): length to estimate at

    Returns
        float: result from filling the values in the model for a 2D lattice
    """
    return A * L ** (3 / 2)


def growth_model_3(L, A):
    """
    a model to fit for the 3D polymer case from [the lecture notes](https://compphys.quantumtinkerer.tudelft.nl/proj2-polymers/#model-polymers-as-a-self-avoiding-random-walk-on-a-lattice)

    Parameters
        A (float): scaling factor for the fit
        L (float): length to estimate at

    Returns
        float: result from filling the values in the model for a 3D lattice
    """
    return A * L ** (6 / 5)


def growth_model_custom(L, A, exp):
    """
    a model to fit for the 2D polymer case from [the lecture notes](https://compphys.quantumtinkerer.tudelft.nl/proj2-polymers/#model-polymers-as-a-self-avoiding-random-walk-on-a-lattice)

    Parameters
        A (float): scaling factor for the fit
        L (float): length to estimate at
        exp (float): exponent (to be optimized)

    Returns
        float: result from filling the values in the model with exponent estimation
    """
    return A * L ** (exp)
