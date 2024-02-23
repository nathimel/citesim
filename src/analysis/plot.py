"""Helper functions for transforming and visualizing atlas data for analysis."""

import hydra
import os

import numpy as np
import pandas as pd
import plotnine as pn

from collections import Counter

import util

from sciterra import Atlas, Cartographer
from sciterra.vectorization.vectorizer import Vectorizer

##############################################################################
# Plotting
##############################################################################

def fields_histogram(atl: Atlas) -> pn.ggplot:
    """Get a (horizontal) historgram with fields on the y-axis, and publication counts on the x axis, corresponding to the distribution of fields of study for publications in the atlas."""
    fields = Counter([field for pub in atl.publications.values() for field in pub.fields_of_study])

    return (
    pn.ggplot(
        data = pd.DataFrame(
            data=[(field, count) for field,count in fields.items()],
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
    min_cpy = 1,
    max_cpy = 100,    
) -> pn.ggplot:
    """Get a histogram of citations per year, given a dataframe of measurements."""

    df_f = data[(data["citations_per_year"] >= min_cpy) & (data["citations_per_year"] <= max_cpy)]

    # Histogram of cpy
    cpy_hist = (
        pn.ggplot(
            # df,
            df_f,
            mapping=pn.aes(x="citations_per_year")
        )
        + pn.geom_histogram(bins=50)
        + pn.scale_x_log10()
        + pn.xlab("Citations per year")
        + pn.ylab(f"Count, total={len(df_f)}")
        + pn.theme(
            axis_title=pn.element_text(size=18)
        )
    )

    return cpy_hist    

# TODO: refactor w above
def density_histogram(
    data: pd.DataFrame,
    min_d = -np.inf,
    max_d = np.inf,
) -> pn.ggplot:
    """Get a histogram of density values, given a dataframe of measurements."""

    df_f = data[(data["density"] >= min_d) & (data["density"] <= max_d)]

    # Histogram of cpy
    cpy_hist = (
        pn.ggplot(
            # df,
            df_f,
            mapping=pn.aes(x="density")
        )
        + pn.geom_histogram(bins=50)
        # + pn.scale_x_log10() # probably don't do this!
        + pn.xlab("Density")
        + pn.ylab(f"Count, total={len(df_f)}")
        + pn.theme(
            axis_title=pn.element_text(size=18)
        )
    )

    return cpy_hist  


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
    # cml_args = [f"--data_fn {df_fn}", f"--plot_dir {plot_dir}"]
    cml_args = [df_fn, save_dir]
    cml_kwargs = [f"--{key} {value}" for key, value in kwargs.items() if value is not None]
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
    con_d: int = 1,
    kernel_size =  16, # TODO: find a principled way of selecting this value.
    fields_of_study = None,
    max_year: int = 2023, # consider 2022
) -> pd.DataFrame:
    """Compute the density, edginess, and citations per year metrics for each publicaation in an atlas w.r.t. a vectorizer and convergence configurations, and return the results in a dataframe.
    
    Args:
        atl: the Atlas to measure

        vectorizer: the Vectorizer to use to compute density and edginess

        con_d: the inverse index for the second axis of the array `atl.history['kernel_size']`, representing the degree of convergence. For details about this array see `sciterra.mapping.cartography.Cartographer.converged_kernel_size`. Default is 1, which means we will filter to all publications that have not changed neighborhoods up to `kernel_size` up until the very last update. If 2, then up to the second to last update, etc.

        kernel_size: the minimum required size of the neighborhood that we will require to not have changed, w.r.t. `cond_d`. Default is 16.

    """
    kernels = atl.history['kernel_size'] # shape `(num_pubs, max_update)`, where `max_update` is typically the total number of updates if this function is called after the atlas has been sufficiently built out.

    # Get all publications that have not changed neighborhoods up to kernel_size for the last con_d updates
    converged_filter = kernels[:, -con_d] >= kernel_size
    ids = np.array(atl.projection.index_to_identifier)
    converged_pub_ids = ids[converged_filter]

    # Optionally filter only to `field_of_study` publications
    # TODO: check this logic is the same as in cartography
    if fields_of_study is not None:
        converged_pub_ids = [id for id in converged_pub_ids if any(fld in atl[id].fields_of_study for fld in atl[id].fields_of_study)]

    # Compute density, edginess
    crt = Cartographer(
        vectorizer=vectorizer,
    )
    measurements = crt.measure_topography(
        atl, 
        ids=converged_pub_ids,
        metrics=["density", "edginess"], 
        kernel_size=kernel_size,
    )

    # Get citations
    citations_per_year = [ 
        atl[id].citation_count / (max_year - atl[id].publication_date.year) if (atl[id].publication_date.year < max_year and atl[id].citation_count is not None) else 0.
        for id in converged_pub_ids
    ]

    # Annotate the center (this feels inefficient, but oh well)
    is_center = [identifier == atl.center for identifier in converged_pub_ids]

    # TODO: why no center in GPT2 Imeletal atlas?
    # Because not converged! ...
    # This suggests that I need to think more about the overall upshot figure, and that it prob shouldn't be GPT2. How can it be that the center pub hasn't converged? 
    # breakpoint()
    if not any(is_center):
        import warnings
        warnings.warn(f"The center publication is not within the set of convered publications.")

    df = pd.DataFrame(
        measurements,
        columns=["density", "edginess"],
    )
    df["citations_per_year"] = citations_per_year
    df["is_center"] = is_center

    df = df[~np.isinf(df["density"])] # drop infs which occur for BOW vectorizer
    # TODO what about other very high densities that result from close to 0?

    df.dropna(inplace=True, )

    return df


