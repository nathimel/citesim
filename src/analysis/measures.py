import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")  # bad


# Helper function for z-scaling
def zscale(df_in: pd.DataFrame, col: str) -> pd.DataFrame:
    result = (df_in[col] - np.nanmean(df_in[col])) / np.nanstd(df_in[col])
    return result


##############################################################################


def bin_measurements(
    df_in: pd.DataFrame, field: str, vectorizer: str, num_quantiles: int
) -> pd.DataFrame:
    """Given a dataframe and a field to subset to, cut density observations into `num_quantiles` (i.e., equal frequency binning) and compute statistics of the resulting citation distribution induced by the density class."""

    # Filter
    df = df_in[df_in["fields_of_study_0"] == field]
    df = df[df["vectorizer"] == vectorizer]

    # Filter within reasonable values
    # First get z-scales for density and citation rates
    for col in ["density", "citations_per_year"]:
        df[f"{col}_z"] = zscale(df, col)

    # For density bin
    density_bin_max = df["density_z"] < 3
    density_bin_min = df["density_z"] > -3
    # For cpy
    cpy_max = df["citations_per_year_z"] < 3
    cpy_min = df["citations_per_year_z"] > -3
    # Apply masks
    df = df[density_bin_max & density_bin_min & cpy_max & cpy_min]

    # cut
    df["density_bin"] = pd.qcut(
        df["density"],
        q=num_quantiles,
    )

    # Group data after binning
    data_bins = df[
        ["log_cpy", "density_bin", "citations_per_year", "references", "year"]
    ]

    # Measure statistics
    with warnings.catch_warnings():
        statistics = []
        for bin_key in sorted(df.density_bin.value_counts().keys()):
            df_bin = data_bins[data_bins["density_bin"] == bin_key]
            statistics.append(
                (
                    np.nanvar(df_bin["log_cpy"].values),
                    np.nanmedian(df_bin["citations_per_year"].values),
                    np.nanmedian(df_bin["references"].values),
                    np.nanmedian(df_bin["year"].values),
                    np.nanmean(df_bin["citations_per_year"].values),
                    np.nanstd(df_bin["citations_per_year"].values),
                    np.nanmean(df_bin["log_cpy"].values),
                    np.nanstd(df_bin["log_cpy"].values),
                )
            )

    # Construct dataframe
    df_result = pd.DataFrame(
        statistics,
        columns=[
            "log_cpy_var",
            "cpy_med",
            "ref_med",
            "year_med",
            "cpy_mean",
            "cpy_std",
            "log_cpy_mean",
            "log_cpy_std",
        ],
    )

    # Annotate by (start of) density bin
    df_result["density_bin"] = [
        float(item.left)
        for item in df.density_bin.value_counts(sort=False, normalize=True).keys()
    ]

    # Annotate z-scales for density and citation rates
    for col in df_result.columns:
        col_z = f"{col}_z"
        df_result[col_z] = zscale(df_result, col)

    # Annotate field and vectorizer
    df_result["field"] = field
    df_result["vectorizer"] = vectorizer

    return df_result
