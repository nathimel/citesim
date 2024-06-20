"""Main driver script to build out regions of scientific literature using similarity-based retrieval."""

import hydra
import os

from omegaconf import DictConfig

import util
from experiment import Experiment

# from analysis.plot import fields_histogram, cpy_histogram, atlas_to_measurements, call_r_2d_histograms

from analysis import plot


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    util.set_seed(config.seed)

    exp = Experiment(config)

    # To disable huggingface parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    atl = exp.atlas

    util.save_plot(
        "fields_histogram.png",
        plot.fields_histogram(atl),
    )

    util.save_plot("years_histogram.png", plot.years_histogram(atl))

    df = plot.atlas_to_measurements(
        atl,
        vectorizer=exp.vectorizer,
        fields_of_study=exp.config.experiment.cartography.required_pub_conditions.fields_of_study,
    )
    df.to_csv("all_data.csv", index=False)

    util.save_plot(
        "cpy_histogram.png",
        plot.cpy_histogram(df),
    )

    util.save_plot(
        "density_histogram.png",
        plot.density_histogram(df),
    )

    # Need to use R to get density plot
    # TOOD: write a function in plot.py to do all this, and just takes args and kwargs about
    df_fn = os.path.join(os.getcwd(), "all_data.csv")
    plot.call_r_2d_histograms(
        df_fn,
        **config.experiment.plot,
    )


if __name__ == "__main__":
    main()
