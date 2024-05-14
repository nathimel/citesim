# Auxilliary R script to run regressions and plot coefficients, called by generate_figures.py


# Imports
library(tidyverse)
library(lme4)
library(modelr)
library(viridis)
library(ggrepel)
library(latex2exp)
library(ggridges)
library(car)
library(lmerTest)
library(lmtest)
library(broom.mixed)
library(jtools)


# Load data
df_analysis <- read_csv("analysis_data/transformed_all_data.csv")


# Regression
getrefs <- function(y, vectorizer_name, print_anova=TRUE) {
  
  df_vectorizer <- df_analysis %>% filter(
    vectorizer == vectorizer_name
  ) %>% mutate(
    # z-scale
    y = .data[[y]],
  ) %>% filter(
    ref_med_z < 3,
    ref_med_z > -3,
  )
  

  model_w_rho <- lmer(
    y ~ ref_med_z + year_med_z + density_bin_z + (1 + density_bin_z | field),
    data=df_vectorizer, 
    control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5))
  )
  
  # if (print_summary) {
  #   print(summary(model_5))
  # }
  
  model_ablated <- lmer(
    y ~ ref_med_z + year_med_z + (1 | field), 
    data=df_vectorizer,
    control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5))
  )

  stats_dir <- "analysis_data/stats/"
  save_fn <- paste(
    y, 
    "_", 
    vectorizer_name, 
    ".txt", 
    sep = ""
  )
  summary_save_fn <- paste(
    stats_dir,
    "summary/",
    save_fn,
    sep = ""
  )
  capture.output(
    summary(model_w_rho),
    file = summary_save_fn
  ) 

  # fc<-file(summary_save_fn)
  # writeLines(c("--------------------------------------------------------------"), fc)
  # close(fc)

  anova_save_fn <- paste(
    stats_dir,
    "anova_ablation/",
    save_fn,
    sep = ""
  )
  capture.output(
    anova(
      model_w_rho,
      model_ablated
    ),
    file = anova_save_fn
  )
  print(paste("Wrote to", anova_save_fn))

  return(model_w_rho)
  
}



# Forest plots

# TODO: clean this up, refactor
for (y in c("cpy_med", "log_cpy_var")) {
  (
   plot_summs(
      getrefs(y, "SciBERT",),
      getrefs(y, "SBERT"),
      getrefs(y, "GPT2"),
      getrefs(y, "Word2Vec"),
      getrefs(y, "BOW"),
      model.names = c("SciBERT", "SBERT", "GPT2", "Word2Vec", "BOW"),
      # model.names = c("SciBERT", "SBERT", "GPT2"),
      colors = "Rainbow",
      point.shape = FALSE,
      robust=TRUE, 
      inner_ci_level = .9
      # plot.distributions = TRUE
    )
    + theme(
      axis.text.x = element_text(size=20),
      axis.text.y = element_blank()
    )

  )
  save_fn <- paste("analysis_data/figures/coefficients/", y, ".png", sep = "")

  ggsave(
    save_fn,
    width = 6,
    height = 4,
  )
  print(paste("Saved an image to", save_fn))
  
}

for (vec in unique(df_analysis$vectorizer)) {
    df_vec <- df_analysis %>% filter(
        vectorizer == vec
    )
    (
        ggplot(
            df_vec,
            aes(
                x=year_med,
                y=density_bin,
            )
        )
        + geom_smooth(
            method = "lm",
            linewidth = 3,
            color = "black",
        )
        + geom_point(
            aes(
                color = field,
            ),
            size = 0.5, 
            alpha = 0.3, 
        )
        + geom_smooth(
            aes(
                color = field,
            ),
            linewidth = 1,
            level = 0.8,
        )

        + scale_x_continuous(
          limits = c(2008, 2016), 
          breaks = seq(2008, 2016, by = 4),
        )
        # + ylim(-3,3)
        + theme_classic()
        # + ggtitle(vec)
        + theme(
            axis.text = element_text(size=30),
            axis.title = element_blank(),
        )
    )

  save_fn <- paste("analysis_data/figures/time/", vec, ".png", sep = "")

  ggsave(
    save_fn,
    width = 8,
    height = 5,
  )
  print(paste("Saved an image to", save_fn))

}


for (vec in unique(df_analysis$vectorizer)) {
    df_vec <- df_analysis %>% filter(
        vectorizer == vec
    ) %>% filter(
    ref_med_z < 3,
    ref_med_z > -3,
    )

    (
        ggplot(
            df_vec,
            aes(
                x=density_bin,
                y=ref_med,
            )
        )
        + geom_smooth(
            method = "lm",
            linewidth = 3,
            color = "black",
        )
        + geom_point(
            aes(
                color = field,
            ),
            size = 0.5, 
            alpha = 0.3, 
        )
        + geom_smooth(
            aes(
                color = field,
            ),
            linewidth = 1,
            level = 0.8,
        )
        + theme_classic()
        # + ggtitle(vec)
        + theme(
            axis.text = element_text(size=30),
            axis.title = element_blank(),
        )
    )

  save_fn <- paste("analysis_data/figures/refs/", vec, ".png", sep = "")

  ggsave(
    save_fn,
    width = 8,
    height = 5,
  )
  print(paste("Saved an image to", save_fn))

}

