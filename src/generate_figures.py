"""Script to generate main figures."""

import sys
import warnings
warnings.filterwarnings("ignore") # bad
import pandas as pd
import numpy as np

from analysis.plot import main_trends_mpl, summary_violins
from util import save_fig
from pathlib import Path

from tqdm import tqdm


# Helper function for z-scaling
def zscale(df_in: pd.DataFrame, col: str) -> pd.DataFrame:
    result = (df_in[col] - np.nanmean(df_in[col])) / np.nanstd(df_in[col])
    return result

##############################################################################

def bin_measurements(df_in: pd.DataFrame, field: str, vectorizer: str, num_bins: int) -> pd.DataFrame:
    """Given a dataframe and a field to subset to, cut density observations into `num_bins` and compute statistics of the resulting citation distribution induced by the density class."""

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
    df['density_bin'] = pd.cut(
        df['density'], 
        bins=np.linspace(
            df['density'].min(), 
            df['density'].max(), 
            num=num_bins+1,
        ),
    )

    # Group data after binning
    data_bins = df[["log_cpy", "density_bin", "citations_per_year", "references", "year"]]

    # Measure statistics
    with warnings.catch_warnings():
        statistics = []
        for bin_key in sorted(df.density_bin.value_counts().keys()):
            df_bin = data_bins[data_bins["density_bin"] == bin_key]
            statistics.append((
                np.nanvar(df_bin["log_cpy"].values),
                np.nanmedian(df_bin["citations_per_year"].values),
                np.nanmedian(df_bin["references"].values),
                np.nanmedian(df_bin["year"].values),
            ))
    
    # Construct dataframe
    df_result = pd.DataFrame(
        statistics,
        columns=[
            "log_cpy_var",
            "cpy_med",
            "ref_med",
            "year_med",
        ],
    )

    # Annotate by (start of) density bin
    df_result["density_bin"] = [float(item.left) for item in df.density_bin.value_counts(sort=False, normalize=True).keys()]

    # Annotate z-scales for density and citation rates
    for col in ["density_bin", "cpy_med", "ref_med", "year_med"]:
        col_z = f"{col}_z"
        df_result[col_z] = zscale(df_result, col)
        
    # Annotate counts and relative frequencies
    df_result["count"] = df.density_bin.value_counts(sort=False, normalize=True).tolist()
    df_result["freq"] = df.density_bin.value_counts(sort=False, normalize=False).tolist()

    # Annotate field and vectorizer
    df_result["field"] = field
    df_result["vectorizer"] = vectorizer

    return df_result

##############################################################################

def transform_by_vectorizer(df_all: pd.DataFrame) -> pd.DataFrame:
    """Transform data for analysis via binned measurements by vectorizer and field.
    
    Args:
        df_all: all observations

    Returns:
        a new dataframe with (concatenated) binned measurements for each vectorizer.
    """
    return pd.concat(
        [
            pd.concat(
                [
                    bin_measurements(
                        df_all[df_all["vectorizer"] == vectorizer], 
                        field,
                        vectorizer,
                        num_bins=200,
                    ) 
                    for field in df_all["fields_of_study_0"].unique()
                ]
            )
            for vectorizer in tqdm(df_all.vectorizer.unique(), "transforming and binning data by vectorizer and field")
        ]
    )


##############################################################################

def main():

    if len(sys.argv) != 2:
        print("Usage: python src/generate_figures.py. PATH_TO_ALL_DATA \nThis script does not use hydra; do not pass overrides.")
        sys.exit(1)

    load_fn = sys.argv[1]
    df_all = pd.read_csv(load_fn)
    analysis_dir = Path(load_fn).parent.absolute()

    # Nan handling across all observations
    df_all["log_cpy"] = np.log10(df_all["citations_per_year"])
    df_all['log_cpy'] = df_all['log_cpy'].replace(-np.inf, np.nan)

    # Main analysis data
    df_analysis = transform_by_vectorizer(df_all)

    # Save
    df_analysis.to_csv(
        analysis_dir / "transformed_all_data.csv",
        index=False,
    )

    # Generate main trends figures
    for vec in df_analysis.vectorizer.unique():
        # Main trends
        fig, _ = main_trends_mpl(df_analysis[df_analysis.vectorizer == vec])
        save_fig(analysis_dir / "figures" / "main_trends" / f"{vec}.png", fig)

        # Violin plots
        fig, _ = summary_violins(df_all[df_all["vectorizer"] == vec], "fields_of_study_0")
        save_fig(analysis_dir / "figures" / "violins" / f"{vec}.png", fig)
    

    # Violin plot aggregating fields; n.b. this doesnt even normalize measurements by field like main_trends does.
    fig, _ = summary_violins(df_all, "vectorizer")
    save_fig(analysis_dir / "figures" / "violins" / f"all.png", fig)

    

if __name__ == "__main__":
    main()