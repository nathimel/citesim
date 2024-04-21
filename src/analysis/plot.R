# Auxilliary R script to plot data, called by analysis.py. If plotnine ever matches ggplot's energy, this file won't need to exist.

library(argparse)
library(dplyr)
library(ggridges)
library(tidyverse)
library(viridis)
library(DescTools)

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

# Optional args for the std filtering
parser$add_argument(
  "--max_cpy_stds",
  nargs=1,
  type="double",
  help="How many standard deviations above the mean to set the max cpy after z-scaling.",
  default=1
)
parser$add_argument(
  "--max_metric_stds",
  nargs=1,
  type="double",
  help="How many standard deviations above the mean to set the max metric value after z-scaling.",
  default=2
)
parser$add_argument(
  "--min_metric_stds",
  nargs=1,
  type="double",
  help="How many standard deviations below the mean to set the max metric value after z-scaling.",
  default=-2
)

args <- parser$parse_args()


df_fn <- args$data_fn
save_dir <- args$save_dir
max_density <- args$max_density
max_cpy_stds <- args$max_cpy_stds
min_metric_stds <- args$min_metric_stds
max_metric_stds <- args$max_metric_stds
log_cpy <- args$log_cpy

# Load data
df <- read_csv(df_fn)

# TODO: if max_density specified, drop rows with greater density values.
# we could groupby and only do this for density, but for consistency we may as well just exclude those values from edginess too
if (is.null(max_density)) {
  max_density <- Inf
}


df <- df %>% filter(
  density <= max_density,
) %>% filter(
  citations_per_year > 0, # We're going to compute log cpys, so need to avoid log0,
)


get_zscaled_filtered <- function(metric) {

  # Filter to 2 stds, for both vars
  metric_z <- paste(metric, "_z", sep="")

  df_z <- df %>% mutate(
    df,
    metric_z = scale(.data[[metric]]),
    cpy_z = scale(citations_per_year),
    references_z = scale(references),
  )
  
  # Add log transform
  df_z <- df_z %>% mutate(
    df_z,
    logcpy = log10( citations_per_year ),
    logcpy_z = log10( cpy_z ),
  )

  # Filter to mean cpys
  # N.B.: sometimes the center will have more than mean!
  # N.B.: physics, but no others, need +1 std for helpful visualization. This should be taken care of by args.
  df_z <- df_z %>%
    filter(
      cpy_z <= max_cpy_stds
    )

  df_zscaled_filtered <- df_z %>% filter(
    (
      metric_z >= min_metric_stds
      & 
      metric_z <= max_metric_stds
    ),
  ) %>% filter(
      references_z <= 3, # might need to outsource to args and config
  )

  return(df_zscaled_filtered)
}

# Plot metric vs. cpy
metric_vs_cpy <- function(metric) {

  df_zf <- get_zscaled_filtered(metric)

  # Get the center to label
  # N.B.: if the python script raised a warning, this dataframe will be empty and no point will be plotted.
  df_center <- df_zf %>% filter(
    is_center == TRUE
  )
  print("DF CENTER:")
  print(df_center)    

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

metric_cpy_basic_regression <- function(metric) {

  df_zf <- get_zscaled_filtered(metric)
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
    + xlab(str_to_title(metric))
    # + ylab("Citations per year")
    + ylab("Citations per year")

    # Local linear regression
    + geom_smooth(color="orange", size=3)

    + geom_point(
      alpha=0.1,
      # color="white",
      size=1,
    )
    
    + ylim(0, NA)
    + theme(
      # axis_title_y=element_blank(),
      axis.title=element_text(size=18),
    )
  )
  return(plot)
}

# Ridges
df_zf <- get_zscaled_filtered("density")
df_zf$density_bin <- cut(
  df_zf$density, 
  breaks = seq(
    min(df_zf$density), 
    max(df_zf$density), 
    length.out = 21
  )
)

joyplot = (
  ggplot(
    df_zf,
    aes(
      x = citations_per_year,
      y = density_bin,
      color = density_bin,
      fill = density_bin,
      alpha = 0.2,
    )
  )
  + geom_density_ridges()
  # + xlim(8,10)
  + scale_color_viridis(discrete = TRUE)
  + scale_fill_viridis(discrete = TRUE)
)
save_fn = paste(save_dir, "/", "density_joyplot.png", sep="")
ggsave(
    save_fn,
    plot=joyplot,
    width=10,
    height=10,
)

# Compute entropy for each bin
entropy_estimator <- function(observations) {
  if (length(observations) == 0) {
    return(NaN)
  }
  
  # Calculate bin width for each dimension
  bin_width <- 20 / 21

  # Compute histogram
  # Compute histogram
  # we could use seq(min_value, max_value, length.out = num_bins + 1),
  # but since we essentially filter to <=10 cpys anyway, just estimate this range evenly
  hist <- hist(observations, breaks = 0:20, plot = FALSE)

  # Compute probabilities for each bin
  bin_probabilities <- hist$counts / length(observations)

  # Calculate entropy for this dimension
  entropy_estimate <- -sum(bin_probabilities * log(bin_probabilities + 1e-16) / bin_width)  # Add small value to avoid log(0)

  return(entropy_estimate)
}

# Now we'll filter to less than 20 cpys; the vast majority of data has less than 10 cpys
df_zf <- df_zf %>% filter(
  citations_per_year <= 20,
)

value_counts <-  count(df_zf, density_bin)
density_bin_counts <- value_counts$n
density_levels <- value_counts$density_bin

cpy_ents <- c()
bin_starts <- c()
for (val in density_levels) {
  # need to do some regexing on the density bin name string
  text <- val
  pattern <- "\\(([-+]?[0-9]*\\.?[0-9]+),"
  numbers <- str_extract_all(text, pattern)
  start <- as.double(lapply(numbers[[1]], function(x) gsub("[(,]", "", x)))

  # now filter and compute entropy
  df_val <- df_zf %>% filter(density_bin == val)
  cpy_values <- unlist(c(df_val["citations_per_year"]))
  h <- entropy_estimator(cpy_values)
  cpy_ents <- append(cpy_ents, h)
  bin_starts <- append(bin_starts, start)
}

df_ent <- tibble(cpy_ents, bin_starts, density_bin_counts)
cpy_ent <- (
  ggplot(
    df_ent,
    aes(
      x=bin_starts,      
      y=density_bin_counts,
      fill=cpy_ents,
    )
  )
  # + geom_point(size=10)
  + geom_col()
  + scale_fill_viridis()
  + labs(fill="citation entropy,\nH(CPY | density_bin)\n")
)
save_fn = paste(save_dir, "/", "citation_entropy_vs_density_bins.png", sep="")
ggsave(
    save_fn,
    plot=cpy_ent,
    width=10,
    height=10,
)

# try visualizing distribution of cpy per density differently,
# this is an alternative to the joy plot.

joyalt <- (
  ggplot(
    df_zf,
  )
  + geom_density(
    aes(
      x=citations_per_year,
      # color = density_bin,
      fill = density_bin,
    ),
    alpha=0.75,    
  )
  + scale_fill_viridis( discrete = TRUE)
)
save_fn = paste(save_dir, "/", "joyalt.png", sep="")
ggsave(
    save_fn,
    plot=joyalt,
    width=10,
    height=10,
)

h_regression <- (
    ggplot(
        df_ent, 
        aes(
            x=bin_starts,
            y=cpy_ents,
        )        
    )
    + geom_point(
        aes(
            size=density_bin_counts,
            alpha=density_bin_counts,
        )
    )
    + geom_smooth(method="lm", color="orange")
    + xlab("density bin")
    + ylab("citation entropy, H(CPY | density_bin)\n")
    + labs(size="count")
    + labs(alpha="count")
    
)
save_fn = paste(save_dir, "/", "entropy_regression.png", sep="")
ggsave(
    save_fn,
    plot=h_regression,
    width=10,
    height=10,
)

# Regression for density and references
refs_regression <- (
  ggplot(
    df_zf,
    aes(x=density, y=references)
  )
  + geom_point(
    alpha=0.2,
  )
  + geom_smooth(color="orange", method="lm")
)
save_fn = paste(save_dir, "/", "references_regression.png", sep="")
ggsave(
  save_fn,
  plot=refs_regression,
  width=10,
  height=10,
)


metric_vs_year <- function(metric) {
  return(
    ggplot(
        df_zf,
        aes(
          x=year,
          y=.data[[metric]],
        )
      )
      + geom_point(alpha=0.2)
      + geom_smooth(color="orange", method="lm")
  )
}

for (metric in c("density", "edginess")) {
  # todo: refactor these into a func and call twice
    save_fn <- paste(save_dir, "/", metric, ".png", sep="")
    ggsave(
        save_fn,
        plot=metric_vs_cpy(metric),
        width=10,
        height=10,
    )

    save_fn <- paste(save_dir, "/", "year_vs_", metric, ".png", sep="")
      ggsave(
        save_fn,
        plot=metric_vs_year(metric),
        width=10,
        height=10,
    )
}

report <- paste("saved plots to", save_dir)
print(report)
