"""Helper functions for estimating efficient frontiers and computing optimality scores."""

import numpy as np
import pandas as pd

from pygmo import non_dominated_front_2d
from scipy import interpolate
from scipy.spatial.distance import cdist

from .measures import zscale


def interpolate_frontier(points: np.ndarray, num: int) -> np.ndarray:
    """Interpolate the frontier points to obtain a dense estimated bound on efficiency.

    Args:
        points: a 2D array of shape `(num_optimal, 2)` representing the non-dominated (Expected Return, Risk) points.

    Returns:
        a 2D array of shape `(num, 2)` representing the interpolated frontier of points.
    """
    # TODO: check this is appropriate
    min_sigma = points.min(axis=0)[0]
    max_sigma = points.max(axis=0)[0]

    pareto_x, pareto_y = list(zip(*points))
    interpolated = interpolate.interp1d(pareto_x, pareto_y, fill_value="extrapolate")

    pareto_costs = list(set(np.linspace(min_sigma, max_sigma, num=num).tolist()))
    pareto_complexities = interpolated(pareto_costs)
    interpolated_points = np.array(list(zip(pareto_costs, pareto_complexities)))
    return interpolated_points


def pareto_min_distances(points: np.ndarray, frontier_points: np.ndarray) -> np.ndarray:
    """Measure the Pareto optimality of each point by measuring its Euclidean closeness to the frontier. The frontier is a line (list of points) interpolated from the pareto points.

    Args:

        points: a 2D array of shape `(num_observations, 2)` representing all (Expected Return, Risk) points.

        pareto_points: a 2D array of shape `(num, 2)` representing the interpolated Pareto frontier of points.

    Returns:

        min_distances: an array of shape `len(points)` Euclidean distances for each point to the closest point on the Pareto frontier.
    """
    # Measure closeness of each language to any frontier point
    distances = cdist(points, frontier_points)
    min_distances = np.min(distances, axis=1)

    # Normalize to 0, 1 because optimality is defined in terms of 1 - dist
    min_distances /= max(min_distances)
    return min_distances


def get_frontier_data(
    df_in: pd.DataFrame, x="log_cpy_std_z", y="log_cpy_mean_z"
) -> pd.DataFrame:
    """Estimate the efficient frontier of the Markowitz bullet using pygmo's non_dominated_front_2d method.

    Args:
        df_in: the data to compute the efficient frontier for

    Returns:
        the subset of df_in representing the efficient frontier.
    """
    # 2D array of ()
    min_points = df_in[[x, y]].values
    min_points[:, 1] *= -1

    dominating_indices = non_dominated_front_2d(min_points)
    return df_in.iloc[dominating_indices]


def annotate_optimality(
    df_binned: pd.DataFrame, risk: str, returns: str
) -> pd.DataFrame:
    """Given a dataframe of binned measurements, return the dataframe annotated with optimality."""
    dominant_data = get_frontier_data(df_binned, risk, returns)
    frontier = interpolate_frontier(dominant_data[[risk, returns]].values, num=5000)
    min_dists = pareto_min_distances(
        df_binned[[risk, returns]].values,
        frontier,
    )
    df_binned["min_distances"] = min_dists
    df_binned["optimality"] = 1 - min_dists

    for col in ["min_distances", "optimality"]:
        col_z = f"{col}_z"
        df_binned[col_z] = zscale(df_binned, col)

    # not the prettiest, but oh well
    df_binned["type"] = "observed"
    dominant_data["type"] = "dominant"
    df_out = pd.concat([df_binned, dominant_data])

    return df_out
