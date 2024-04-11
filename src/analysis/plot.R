# Auxilliary R script to plot data, called by analysis.py. If plotnine ever matches ggplot's energy, this file won't need to exist.

library(tidyverse)
library(dplyr)
library(viridis)
library(argparse)


# create parser object
parser <- ArgumentParser()

parser$add_argument(
  "data_fn",
  nargs=1, 
  help="Path to CSV file containing atlas topography measurements."
)

parser$add_argument(
  "save_dir",
  nargs=1,
  help="Path to directory to save plots."
)

parser$add_argument(
  "--max_density",
  type="double",
  help="Drop all observations from plotting density for values of density greater than this value."
)

parser$add_argument(
  "--log_cpy",
  nargs=1,
  type="logical",
  help="Whether to plot the y-axis, citations per year, on a log scale.",
  default=FALSE
)

# TODO: add optional args for the std filtering

args <- parser$parse_args()


df_fn <- args$data_fn
save_dir <- args$save_dir
max_density <- args$max_density
log_cpy <- args$log_cpy

# Load data
df <- read_csv(df_fn)

# TODO: if max_density specified, drop rows with greater density values.
# we could groupby and only do this for density, but for consistency we may as well just exclude those values from edginess too
if (is.null(max_density)) {
  max_density <- Inf
}

df <- df %>% filter(
  density <= max_density
)

# Plot metric vs. cpy
metric_vs_cpy <- function(metric) {
  # Filter to 2 stds, for both vars
  metric_z <- paste(metric, "_z", sep="")

  df_z <- df %>% mutate(
    df,
    metric_z = scale(.data[[metric]]),
    cpy_z = scale(citations_per_year),
  )
  
  # Add log transform
  df_z <- df_z %>% mutate(
    df_z,
    logcpy = log10( citations_per_year ),
    logcpy_z = log10( cpy_z ),
  )

  # Get the center to label
  # N.B.: if the python script raised a warning, this dataframe will be empty and no point will be plotted.
  df_center <- df_z %>% filter(
    is_center == TRUE
  )
  # print("DF CENTER:")
  # print(df_center)  
  
  # Filter to mean cpys
  # TODO: sometimes the center will have more than mean!
  # Maybe do +1 std?
  # N.B.: physics, but no others, need +1 std for helpful visualization. This suggests we should outsource to config. 
  df_z <- df_z %>%
    filter(
      cpy_z <= 1
    )

  df_zf <- df_z %>% filter(
    (
      metric_z >= -2
      & 
      metric_z <= 2
    ),
  )

  y <- "citations_per_year"
  if (log_cpy) {
    y <- "logcpy"
  }
  
  plot = (
    ggplot(
      df_zf,
      mapping=aes(
        # x=density_z, 
        x=.data[[metric]], # NOTE that we filter by z-scale, but can still plot the orig values.
        # y=citations_per_year
        y=.data[[y]],
      )
    )
    + geom_density_2d_filled(
      contour_var = "ndensity",
      # alpha=0.2,
    )
    + scale_fill_viridis(option = "viridis", discrete = TRUE)
    # + xlab("Density z-scaled")
    + xlab(str_to_title(metric))
    # + ylab("Citations per year")
    + ylab("Citations per year")

    # Local linear regression
    + geom_smooth(color="orange", size=2, method="loess", span=.3)

    # Running binned median
    + stat_summary_bin(fun = "median", geom = "smooth", bins = 10)

    + geom_point(
      alpha=0.05,
      color="white",
      size=1,
    )
    
    # Annotate the center
    + geom_label(
      data=df_center,
      mapping=aes(label="center publication"),
      nudge_y = -.2,
    )
    + geom_point(
      data=df_center,
      # shape=4,
      size=6,
      color="red",
    )
    
    + ylim(0, NA)
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

report <- paste("saved plots to", save_dir)
print(report)
