import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import scipy.optimize as opt


def plot_gyration(
    lengths: NDArray[np.int64],
    r2: NDArray[np.float64],
    weights: NDArray[np.float64],
    dim: int,
    alive: bool,
    max_step: int,
):
    mean_r2, error_r2 = analytical_error(r2, weights, alive)

    fig, ax = plt.subplots()

    ax.set_title("Length dependent radius of Gyration")
    ax.set_xlabel(r"L ($\sigma$)")
    ax.set_ylabel(r"Radius of Gyration ($\sigma^2$)")

    ax.plot(lengths, mean_r2, label="gyration", color="C0")
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
        )
    elif dim == 3:
        opt_params, _ = opt.curve_fit(growth_model_3, lengths, mean_r2)
        ax.plot(
            lengths,
            opt_params[0] * lengths ** (6 / 5),
            label=f"best fit: ${opt_params[0]:.03f} L^{{6/5}}$",
        )

    # Show number of polymers on secondary y-axis
    ax_right = ax.twinx()
    ax_right.set_ylabel("amount of polymers")
    ax_right.set_yscale("log")
    ax_right.plot(lengths, np.sum(alive[:, :max_step], axis=0), color="gray", alpha=0.5)

    ax.legend()
    plt.tight_layout()
    fig.savefig("gyration.png")
    plt.close()


def plot_end_to_end(
    lengths: NDArray[np.int64],
    r2: NDArray[np.float64],
    weights: NDArray[np.float64],
    dim: int,
    alive: bool,
    max_step: int,
):
    mean_r2, error_r2 = analytical_error(r2, weights, alive)

    fig, ax = plt.subplots()

    ax.set_title("Length dependent end-to-end distance")
    ax.set_xlabel(r"L ($\sigma$)")
    ax.set_ylabel(r"end to end dist ($\sigma^2$)")

    ax.plot(lengths, mean_r2, label="end-to-end", color="C0")
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
        )
    elif dim == 3:
        opt_params, _ = opt.curve_fit(growth_model_3, lengths, mean_r2)
        ax.plot(
            lengths,
            opt_params[0] * lengths ** (6 / 5),
            label=f"best fit: ${opt_params[0]:.03f} L^{{6/5}}$",
        )

    # Show number of polymers on secondary y-axis
    ax_right = ax.twinx()
    ax_right.set_ylabel("amount of polymers")
    ax_right.set_yscale("log")
    ax_right.plot(lengths, np.sum(alive[:, :max_step], axis=0), color="gray", alpha=0.5)

    ax.legend()
    plt.tight_layout()
    fig.savefig("end_to_end.png")
    plt.close()


def analytical_error(r2, w, alive):
    N_temp, max_step = r2.shape
    N = np.sum(alive[:, :max_step], axis=0)
    r2_mean = np.sum(w * r2, axis=0) / np.sum(w, axis=0)
    numerator = np.sum((w / np.max(w)) ** 2 * (r2 - r2_mean) ** 2, axis=0)
    denominator = (np.sum(w / np.max(w), axis=0)) ** 2
    error = np.sqrt((N / (N - 1)) * (numerator / denominator))
    return r2_mean, error


def growth_model(L, A):
    return A * L ** (3 / 2)


def growth_model_3(L, A):
    return A * L ** (6 / 5)

