"""Main driver script to build out regions of scientific literature using similarity-based retrieval."""

import hydra
import os

from omegaconf import DictConfig

import util
from experiment import Experiment
from analysis.plot import fields_histogram, cpy_histogram, atlas_to_measurements


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    util.set_seed(config.seed)

    exp = Experiment(config)

    # To disable huggingface parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    atl = exp.atlas

    util.save_plot(
        "fields_histogram.png",
        fields_histogram(atl),
    )

    df = atlas_to_measurements(
        atl,
        vectorizer=exp.vectorizer,
        fields_of_study=exp.config.experiment.cartography.required_pub_conditions.fields_of_study
    )
    df.to_csv("all_data.csv", index=False)

    util.save_plot(
        "cpy_histogram.png",
        cpy_histogram(df),
    )

    # Need to use R to get density plot
    df_fn = os.path.join(os.getcwd(), "all_data.csv")
    plot_dir = os.path.join(os.getcwd())
    chdir = f"cd {hydra.utils.get_original_cwd()}"
    ex = f"Rscript src/analysis/plot.R {df_fn} {plot_dir}"
    command = f"{chdir}; {ex}"
    os.system(f"echo {command}")
    os.system(command)


if __name__ == "__main__":
    main()
