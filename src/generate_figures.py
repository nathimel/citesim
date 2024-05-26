"""Script to generate main figures."""

import sys
import warnings
warnings.filterwarnings("ignore") # bad
import pandas as pd
import numpy as np

from analysis.plot import main_trends_mpl, summary_violins, tradeoffs_faceted, tradeoffs_aggregated
from analysis.efficiency import annotate_optimality
from analysis.measures import bin_measurements

from util import save_fig, save_plot
from pathlib import Path

from tqdm import tqdm



def transform_by_vectorizer(df_all: pd.DataFrame, risk: str, returns: str,) -> pd.DataFrame:
    """Transform data for analysis via binned measurements by vectorizer and field.
    
    Args:
        df_all: all observations

    Returns:
        a new dataframe with (concatenated) binned measurements for each vectorizer and measured optimality.
    """
    return pd.concat(
        [
            pd.concat(
                [
                    annotate_optimality(
                        bin_measurements(
                            df_all[df_all["vectorizer"] == vectorizer], 
                            field,
                            vectorizer,
                            num_quantiles=100,
                        ),
                        risk,
                        returns,
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

    # Set the axes and color variables for tradeoffs
    risk = "log_cpy_std_z"
    returns = "log_cpy_mean_z"
    color = "density_bin_z"

    # Main analysis data
    df_analysis = transform_by_vectorizer(df_all, risk, returns)

    # Save
    df_analysis.to_csv(
        analysis_dir / "transformed_all_data.csv",
        index=False,
    )

    # Generate main trends figures
    for vec in df_analysis.vectorizer.unique():
        df_vec = df_analysis[df_analysis.vectorizer == vec]

        # Tradeoffs by field
        save_plot(
            analysis_dir / "figures" / "tradeoffs" / "faceted" / f"{vec}.png",
            tradeoffs_faceted(df_vec, risk, returns, color),
        )
        # Tradeoffs aggregating field
        save_plot(
            analysis_dir / "figures" / "tradeoffs" / "all" / f"{vec}.png", 
            tradeoffs_aggregated(
                annotate_optimality(     # re annotate globally
                    df_vec, risk, returns,
                ),
                risk, returns, color,
            ),
        )

        # Individual trends
        fig, _ = main_trends_mpl(df_vec)
        save_fig(analysis_dir / "figures" / "main_trends" / f"{vec}.png", fig)

        # Violin plots
        fig, _ = summary_violins(df_all[df_all["vectorizer"] == vec], "fields_of_study_0")
        save_fig(analysis_dir / "figures" / "violins" / f"{vec}.png", fig)


    # Violin plot aggregating fields
    fig, _ = summary_violins(df_all, "vectorizer")
    save_fig(analysis_dir / "figures" / "violins" / f"all.png", fig)

    

if __name__ == "__main__":
    main()