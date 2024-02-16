# Auxilliary R script to plot data, called by analysis.py. If plotnine ever matches ggplot's energy, this file won't need to exist.

library(tidyverse)
library(lme4)
library(modelr)
library(viridis)
library(ggrepel)
library(latex2exp)

# Parse cml args

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 2) {
  cat("Usage: Rscript my_script.R <path-to-analysis-csv> <path-to-save-plots> \n")
  q("no", status = 1)
}
df_fn <- args[1] # abs path
save_dir <- args[2]



# Load data
df <- read_csv(df_fn)

# Plot metric vs. cpy
metric_vs_cpy <- function(metric) {
  # Filter to 2 stds, for both vars
  metric_z <- paste(metric, "_z", sep="")

    df_z <- df %>% mutate(
        df,
        metric_z = scale(.data[[metric]]),
        cpy_z = scale(citations_per_year),
    )

    # Filter to mean cpys
    df_z <- df_z %>%
        filter(
            cpy_z <= 0
        )


  df_zf <- df_z %>%
    filter(
      (
        metric_z >= -2 
        & 
        metric_z <= 2
      ),
    )
  plot = (
    ggplot(
      df_zf,
      mapping=aes(
        # x=density_z, 
        x=.data[[metric]], # NOTE that we filter by z-scale, but can still plot the orig values.
        y=citations_per_year
      )
    )
    + geom_density_2d_filled(
      contour_var = "ndensity",
      # alpha=0.2,
    )
    + scale_fill_viridis(option = "viridis", discrete = TRUE)
    # + xlab("Density z-scaled")
    + xlab(str_to_title(metric))
    + ylab("Citations per year")
    + geom_smooth(color="orange", size=2, method="loess", span=.3)
    + geom_point(
      alpha=0.05,
      color="white",
      size=1,
    )
    + theme(
      # axis_title_y=element_blank(),
      axis.title=element_text(size=18),
    )
  )
  return(plot)
}

for (metric in c("density", "edginess")) {
    save_fn <- paste(save_dir, "/", metric, ".png", sep="")
    ggsave(
        save_fn,
        plot=metric_vs_cpy(metric),
        width=10,
        height=10,
    )
}