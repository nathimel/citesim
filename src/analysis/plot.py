"""Helper functions for transforming and visualizing atlas data for analysis."""

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




##############################################################################
# Data transforms
##############################################################################

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

    # get citations
    citations_per_year = [ 
        atl[id].citation_count / (max_year - atl[id].publication_date.year) if (atl[id].publication_date.year < max_year and atl[id].citation_count is not None) else 0.
        for id in converged_pub_ids
    ]

    df = pd.DataFrame(
        measurements,
        columns=["density", "edginess"],
    )
    df["citations_per_year"] = citations_per_year

    df = df[~np.isinf(df["density"])] # drop infs which occur for BOW vectorizer

    df.dropna(inplace=True, )     

    return df

