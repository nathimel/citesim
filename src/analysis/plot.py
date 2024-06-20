"""Helper functions for transforming and visualizing atlas data for analysis."""

import hydra
import os

import numpy as np
import pandas as pd
import plotnine as pn

from collections import Counter


from sciterra import Atlas, Cartographer
from sciterra.vectorization.vectorizer import Vectorizer
from sciterra.mapping.tracing import search_converged_ids


##############################################################################
# Plotting
##############################################################################


def fields_histogram(atl: Atlas) -> pn.ggplot:
    """Get a (horizontal) historgram with fields on the y-axis, and publication counts on the x axis, corresponding to the distribution of fields of study for publications in the atlas."""
    fields = Counter(
        [field for pub in atl.publications.values() for field in pub.fields_of_study]
    )

    return (
        pn.ggplot(
            data=pd.DataFrame(
                data=[(field, count) for field, count in fields.items()],
                columns=["field", "count"],
            ),
            mapping=pn.aes(x="field", y="count"),
        )
        # + pn.scale_y_continuous(trans = "log10")
        + pn.geom_col()
        + pn.coord_flip()
    )


def cpy_histogram(
    data: pd.DataFrame,
    min_cpy=1,
    max_cpy=100,
) -> pn.ggplot:
    """Get a histogram of citations per year, given a dataframe of measurements."""

    df_f = data[
        (data["citations_per_year"] >= min_cpy)
        & (data["citations_per_year"] <= max_cpy)
    ]

    # Histogram of cpy
    cpy_hist = (
        pn.ggplot(
            # df,
            df_f,
            mapping=pn.aes(x="citations_per_year"),
        )
        + pn.geom_histogram(bins=50)
        + pn.scale_x_log10()
        + pn.xlab("Citations per year")
        + pn.ylab(f"Count, total={len(df_f)}")
        + pn.theme(axis_title=pn.element_text(size=18))
    )

    return cpy_hist


# TODO: refactor w above
def density_histogram(
    data: pd.DataFrame,
    min_d=-np.inf,
    max_d=np.inf,
) -> pn.ggplot:
    """Get a histogram of density values, given a dataframe of measurements."""

    df_f = data[(data["density"] >= min_d) & (data["density"] <= max_d)]

    # Histogram of cpy
    cpy_hist = (
        pn.ggplot(
            # df,
            df_f,
            mapping=pn.aes(x="density"),
        )
        + pn.geom_histogram(bins=50)
        # + pn.scale_x_log10() # probably don't do this!
        + pn.xlab("Density")
        + pn.ylab(f"Count, total={len(df_f)}")
        + pn.theme(axis_title=pn.element_text(size=18))
    )

    return cpy_hist


# TODO: refactor w above
def years_histogram(
    atl: Atlas,
) -> pn.ggplot:
    """Get a histogram of the years of publication."""

    years = Counter([pub.publication_date.year for pub in atl.publications.values()])

    return (
        pn.ggplot(
            data=pd.DataFrame(
                data=[(year, count) for year, count in years.items()],
                columns=["year", "count"],
            ),
            mapping=pn.aes(x="year", y="count"),
        )
        + pn.geom_col()
    )


# R backend
# TODO: consider adding the identity of pubs in the data, so we can visualize where the center is in the distribution.
def call_r_2d_histograms(
    df_fn: str,
    save_dir: str = None,
    **kwargs,
) -> None:
    """Run the backend R script `src/analysis/plot.R`, which created histograms of metrics vs. cpy, filled by density of data points. This plot is not available in plotnine.

    Args:
        df_fn: the absolute path to the CSV containing the measurements resulting from `atlas_to_measurements`.

        save_dir: the absolute path to the directory to save the plots. Default is the parent directory of `df_fn`.
    """
    if save_dir is None:
        save_dir = os.path.dirname(df_fn)

    # Process args
    cml_args = [df_fn, save_dir]
    cml_kwargs = [
        f"--{key} {value}" for key, value in kwargs.items() if value is not None
    ]
    all_cml_args = " ".join(cml_args + cml_kwargs)

    # Run the R script
    chdir = f"cd {hydra.utils.get_original_cwd()}"
    run_rscript = f"Rscript src/analysis/plot.R {all_cml_args}"
    command = f"{chdir}; {run_rscript}"
    os.system(f"echo '{command}'")
    os.system(command)


##############################################################################
# Data transforms
##############################################################################


# TODO: should we move this into a separate file?
def atlas_to_measurements(
    atl: Atlas,
    vectorizer: Vectorizer,
    converged_pub_ids: list[str] = None,
    num_pubs_added: int = 1000,
    kernel_size=16,  # TODO: find a principled way of selecting this value.
    fields_of_study=None,
    max_year: int = 2020,  # consider 2020, since that is when exp growth falls
    min_year: int = 2000,
) -> pd.DataFrame:
    """Compute the density, edginess, and citations per year metrics for each publication in an atlas w.r.t. a vectorizer and convergence configurations, and return the results in a dataframe.

    Args:
        atl: the Atlas to measure

        vectorizer: the Vectorizer to use to compute density and edginess

        converged_pub_ids: the identifiers of publications to consider for analysis. By default is `None`, and will be inferred on the bases of the next two parameters.

        num_pubs_added: propagated to `search_converged_ids`

        kernel_size: propagated to `search_converged_ids`
    """

    if converged_pub_ids is None:
        converged_pub_ids = search_converged_ids(
            atl,
            num_pubs_added=num_pubs_added,
            kernel_size=kernel_size,
        )

    # Optionally filter only to `field_of_study` publications
    if fields_of_study is not None:
        converged_pub_ids = [
            id
            for id in converged_pub_ids
            if any(fld in atl[id].fields_of_study for fld in atl[id].fields_of_study)
        ]

    # Compute density, edginess
    crt = Cartographer(
        vectorizer=vectorizer,
    )
    topo_metrics = ["density", "edginess"]
    measurements = crt.measure_topography(
        atl,
        ids=converged_pub_ids,
        metrics=topo_metrics,
        kernel_size=kernel_size,
    )

    # Count references.
    references = [len(atl[id].references) for id in converged_pub_ids]

    valid_year = lambda year: year < max_year and year > min_year
    # valid_year = lambda year: year == 2010 # robustness check that trends hold for specific years

    # Get citations
    citations_per_year = [
        atl[id].citation_count / (max_year - atl[id].publication_date.year)
        if (
            valid_year(atl[id].publication_date.year)
            and atl[id].citation_count is not None
        )
        else np.nan  # prev set else to 0, but this is less ambiguous
        for id in converged_pub_ids
    ]

    # Get year
    years = [atl[id].publication_date.year for id in converged_pub_ids]

    # Annotate the center (this feels inefficient, but oh well)
    is_center = [identifier == atl.center for identifier in converged_pub_ids]

    if not any(is_center):
        import warnings

        warnings.warn(
            f"The center publication is not within the set of converged publications."
        )

    df = pd.DataFrame(
        measurements,
        columns=topo_metrics,
    )
    df["references"] = references
    df["citations_per_year"] = citations_per_year
    df["is_center"] = is_center
    # Annotate with ids, which can be helpful for copying atlases
    df["identifier"] = converged_pub_ids
    # Annotate with year of publication
    df["year"] = years

    df = df[~np.isinf(df["density"])]  # drop infs which occur for BOW vectorizer
    # TODO what about other very high densities that result from close to 0?

    df.dropna(
        inplace=True,
    )

    print(f"There are {len(df)} total observations after filtering.")

    return df


import scipy
import matplotlib.pyplot as plt
import warnings
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def main_trends_mpl(df_plot: pd.DataFrame) -> tuple[Figure, Axes]:
    """Main trend doubleplot with binned density (z-scaled) on x axis and median cpy and log cpy variance on y axis.

    Args:
        df_plot: dataframe containing columns ['density_bin_z', 'cpy_med', 'log_cpy_var', 'field']. This should only contain measurements filtered to a specific vectorizer, and already filtered between reasonable z-scores (e.g., -3 and 3).

    Returns:
        a pair (fig, ax) corresponding to matplotlib Figure and Axes for the plot.

    """

    rows = ["median", "variance"]
    n_rows = 2

    facecolor = np.array([235, 235, 235]) / 256.0
    fig = plt.figure(figsize=(12, 5 * n_rows), facecolor=facecolor)

    gs = gridspec.GridSpec(n_rows, 1)
    gs.update(hspace=0.05, wspace=0.001)

    x_variable = "density_bin_z"
    y_variable_maps = {
        "median": "log_cpy_mean_z",
        "variance": "log_cpy_std_z",
    }

    # For each trend
    for row_idx, row in enumerate(rows):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax = plt.subplot(
                gs[row_idx, 0],
            )

        y_variable = y_variable_maps[row]
        x_observations = df_plot[x_variable]
        y_observations = df_plot[y_variable]

        # N.B.: global scatter, median, percentiles not really well formed across fields, since citation rates are field specific.

        # GLobal

        bin_edges = np.linspace(
            x_observations.min(),
            x_observations.max(),
            10,
        )

        xs = bin_edges[:-1] + 0.33 * (bin_edges[1] - bin_edges[0])

        #######################################
        # Fields
        #######################################

        fields = df_plot.field.unique()
        cmap = plt.cm.jet  # define the colormap

        for field in fields:
            df_field = df_plot[df_plot["field"] == field]

            x_observations_field = df_field[x_variable]
            y_observations_field = df_field[y_variable]

            bin_edges = np.linspace(
                x_observations_field.min(),
                x_observations_field.max(),
                6,
            )

            xs = bin_edges[1:]

            # Binned median
            median, bin_edges, _ = scipy.stats.binned_statistic(
                x_observations_field, y_observations_field, "median", bins=bin_edges
            )
            upper_fn = lambda y: np.nanpercentile(y, 68)
            upper, _, _ = scipy.stats.binned_statistic(
                x_observations_field, y_observations_field, upper_fn, bins=bin_edges
            )
            lower_fn = lambda y: np.nanpercentile(y, 32)
            lower, _, _ = scipy.stats.binned_statistic(
                x_observations_field, y_observations_field, lower_fn, bins=bin_edges
            )
            ax.fill_between(
                xs,
                lower,
                upper,
                color="gray",
                alpha=0.15,
            )

            ax.scatter(
                x_observations_field,
                y_observations_field,
                alpha=0.4,
                s=6,
            )

            ax.plot(
                xs,
                median,
                linewidth=4,
                label=field,
            )

        #######################################
        # Label rows
        #######################################
        if row == "median":
            # ax.set_ylim(0,11)
            ax.set_xlim(-2.75, 2.75)

            ax.set_ylabel(r"Mean, $\mu_{\log(cpy)}$", fontsize=16)

        if row == "variance":
            # ax.set_ylim(0.1, 0.8,)
            ax.set_xlim(-2.75, 2.75)

            ax.set_ylabel(r"Std, $\sigma_{\log(cpy)}$", fontsize=16)

        # Customize ticks
        ax.tick_params(right=True, labelright=True)
        if not ax.get_subplotspec().is_last_row():
            ax.tick_params(axis="x", labelbottom=False)

    axbox = ax.get_position()
    ax.legend(
        prop={
            "size": 14,
        },
        ncol=1,
        # loc = 'upper right',
        loc=(axbox.x0 + -0.12, axbox.y0 + 1.23),
        # loc = (axbox.x0 + 0.6, axbox.y0 + 1.23),
        framealpha=0.5,
    )

    ax.set_xlabel(
        r"Density (z-score) of previously existing research, $\rho$", fontsize=16
    )

    # plt.tight_layout()

    return (fig, ax)


import seaborn as sns
import matplotlib.transforms
from matplotlib import patheffects


def summary_violins(
    df_all: pd.DataFrame, categorization: str, quantile: float = 0.25
) -> tuple[Figure, Axes]:
    """Get a multi violin plot, by field or vectorizer.

    Args:
        df_all: all observations, before any transforming.

        categorization: one of {"fields_of_study_0", "vectorizer"}

        quantile: the upper and lower quantile to compare. E.g., 0.5 is median, so lower 50% and 50% will be compared

    Returns:
        a pair (fig, ax) corresponding to matplotlib Figure and Axes for the plot.
    """
    percent = int(100 * quantile)

    # Transform data
    category_names = sorted(df_all[categorization].unique())
    dfs_category_quantiles = []
    for category in category_names:
        df_cat = df_all[df_all[categorization] == category]

        # Create density categories
        upper_quantile = df_cat["density"].quantile(1 - quantile)
        lower_quantile = df_cat["density"].quantile(quantile)
        df_cat["density_cat"] = "center"
        df_cat.loc[
            df_cat["density"] < lower_quantile, "density_cat"
        ] = f"lower {percent}%"
        df_cat.loc[
            df_cat["density"] > upper_quantile, "density_cat"
        ] = f"upper {percent}%"
        df_cat = df_cat.loc[df_cat["density_cat"] != "center"]
        df_cat["density_cat"] = df_cat["density_cat"].astype("category")

        # Annotate category
        df_cat[categorization] = category

        dfs_category_quantiles.append(df_cat)

    # Combine
    df_quantiles = pd.concat(dfs_category_quantiles)
    # Correct duplicates
    df_quantiles = df_quantiles.reset_index()

    # Start building figure
    fig = plt.figure(figsize=(len(category_names) * 2.5, 6))
    ax = plt.gca()
    ax.set_title("Density")

    sns.violinplot(
        ax=ax,
        data=df_quantiles,
        x=categorization,
        y="log_cpy",
        hue="density_cat",
        split=True,
        inner="quart",
        dodge=True,
        gap=0,
    )

    df_by_cat = df_quantiles.groupby(categorization)
    fraction_changes = []
    fraction_std_changes = []
    for i, key in enumerate(category_names):
        # Get the group
        try:
            df_category = df_by_cat.get_group(key)
        except:
            breakpoint()
        n = df_category.shape[0]

        # Median change
        df_category_by_density = df_category.groupby("density_cat")
        med_cpy = 10.0 ** df_category_by_density["log_cpy"].median()
        fraction_change = med_cpy[f"upper {percent}%"] / med_cpy[f"lower {percent}%"]
        fraction_changes.append(fraction_change)
        median_change_str = (
            rf"$c_{{>{percent}}} = " f"{fraction_change:.2f}" rf"c_{{<{percent}}}$"
        )

        # Median change in width
        std_cpy = 10.0 ** df_category_by_density["log_cpy"].std()
        fraction_std_change = (
            std_cpy[f"upper {percent}%"] / std_cpy[f"lower {percent}%"]
        )
        fraction_std_changes.append(fraction_std_change)
        std_change_str = (
            rf"$\sigma_{{>{percent}}} = "
            f"{fraction_std_change:.2f}"
            rf"\sigma_{{<{percent}}}$"
        )

        text = ax.annotate(
            text=f"n={n}\n" + median_change_str + "\n" + std_change_str,
            xy=(i, 1),
            xycoords=matplotlib.transforms.blended_transform_factory(
                ax.transData,
                ax.transAxes,
            ),
            xytext=(0, -5),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=14,
        )
        text.set_path_effects(
            [patheffects.Stroke(linewidth=3, foreground="w"), patheffects.Normal()]
        )

    ax.legend(prop=dict(size=14))
    legend = ax.get_legend()
    legend.set_title("Density percentile", prop=dict(size=14))
    legend.set_loc("upper center")
    legend.set_bbox_to_anchor((0.56, 0.84))
    legend.set_alignment("left")

    return (fig, ax)


def tradeoffs_faceted(
    df_plot: pd.DataFrame, risk: str, returns: str, color: str
) -> pn.ggplot:
    df_all = df_plot[df_plot["type"] == "observed"]
    df_dominant = df_plot[df_plot["type"] == "dominant"]

    return (
        pn.ggplot(
            df_all,
            pn.aes(
                x=risk,
                y=returns,
            ),
        )
        + pn.geom_point(
            pn.aes(color=color),
            size=4,
            alpha=0.8,
        )
        # + pn.geom_line(
        #     df_dominant,
        #     color = "black",
        #     size = 2,
        #     # alpha = 0.2,
        #     linetype = "dashed",
        # )
        + pn.facet_wrap("field")
        # + pn.theme_classic()
        + pn.scale_color_continuous("cividis")
        + pn.labs(color="Density, $\\rho$")
        + pn.xlab("Risk, $\sigma_{\log cpy}$")
        + pn.ylab("Return, $\mu_{\log cpy}$")
        + pn.theme(
            # Axis font
            axis_title=pn.element_text(size=24),
            axis_text=pn.element_text(size=18),
            strip_text=pn.element_text(size=18),
        )
    )


# Could refactor with above
def tradeoffs_aggregated(
    df_plot: pd.DataFrame, risk: str, returns: str, color: str
) -> pn.ggplot:
    df_all = df_plot[df_plot["type"] == "observed"]
    df_dominant = df_plot[df_plot["type"] == "dominant"]

    return (
        pn.ggplot(
            df_all,
            pn.aes(
                x=risk,
                y=returns,
                # color="density_bin_z", #
            ),
        )
        + pn.geom_line(
            df_dominant,
            color="black",
            size=4,
            # alpha = 0.2,
            linetype="dashed",
        )
        + pn.geom_point(
            pn.aes(color=color),
            size=6,
            alpha=0.8,
        )
        + pn.scale_color_continuous("cividis")
        + pn.labs(color="Density, $\\rho$")
        + pn.xlab("Risk, $\sigma_{\log cpy}$")
        + pn.ylab("Return, $\mu_{\log cpy}$")
        + pn.theme_classic()
        + pn.theme(
            # Axis font
            axis_title=pn.element_text(size=24),
            axis_text=pn.element_text(size=18),
        )
    )
