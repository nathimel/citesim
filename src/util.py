import numpy as np
from plotnine import ggplot
from sciterra import Atlas
from sciterra.mapping.tracing import search_converged_ids

from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pseudo random
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def set_seed(seed: int) -> None:
    """Sets various random seeds."""
    np.random.seed(seed)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# File-writing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def save_plot(fn: str, plot: ggplot, width=10, height=10, dpi=300) -> None:
    """Save a plot with some default settings."""
    plot.save(fn, width=width, height=height, dpi=dpi, verbose=False)
    print(f"Saved a plot to {fn}")


def save_fig(fn: str, fig: Figure, dpi=300) -> None:
    fig.savefig(fn, dpi=dpi)
    plt.close(fig)
    print(f"Saved a plot to {fn}")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Expansion helper function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def converged_pubs_convergence_func(atl: Atlas, size: int, *args, **kwargs) -> bool:
    """Calls `analyze.plot.atlas_to_measurements` and returns True if the resulting dataframe has `size` total rows.


    Args:
        atl: the atlas to pass to `atlas_to_measurements`

        size: the length that the resulting dataframe returned from `atlas_to_measurements` must be greater than or equal to in order to return True

        args: propagated to `atlas_to_measurements`

        kwargs: propagated to `atlas_to_measurements`
    """
    # return len(atlas_to_measurements(atl, *args, **kwargs)) >= size
    return len(search_converged_ids(atl, *args, **kwargs)) >= size
