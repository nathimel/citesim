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
getrefs <- function(y, vectorizer_name, print_summary=TRUE) {
  
  df_vectorizer <- df_analysis %>% filter(
    vectorizer == vectorizer_name
  ) %>% mutate(
    # z-scale
    y = .data[[y]],
    year_med_z = scale(year_med),
    ref_med_z = scale(ref_med),
  ) %>% filter(
    ref_med_z < 3,
    ref_med_z > -3,
  )
  
  # Model 5: 
  label_5 <- paste(
    # "y_z ~ year_med_z + density_bin_z + (1 | field)"
    # "y_z ~ ref_med_z + year_med_z + density_bin_z + (1 | field)"
    # "y_z ~ ref_med_z + year_med_z + density_bin_z + (1 + density_bin_z + ref_med_z | field)"
    "y ~ ref_med_z + year_med_z + density_bin_z + (1 + density_bin_z | field)"
  )
  fm_5 <- as.formula(label_5)
  model_5 <- lmer(
    fm_5, 
    data=df_vectorizer, 
    control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5))
  )
  
  if (print_summary) {
    print(summary(model_5))    
  }
  
  # model_ref_rho <- lmer(
  #   ref_med_z ~ density_bin_z + (1 + density_bin_z | field), 
  #   data=df_vectorizer,
  #   control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5))
  # )
  # 
  # print(summary(model_ref_rho))

  return(model_5)
  
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
      axis.text.x = element_text(size=20)
    )

  )
  save_fn <- paste("analysis_data/figures/coefficients/", y, ".png", sep = "")

  ggsave(
    save_fn
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
            alpha = 0.2, 
        )
        + geom_smooth(
            aes(
                color = field,
            ),
            linewidth = 1,
        )
        + theme_classic()
        + ggtitle(vec)
        + theme(
            axis.text = element_text(size=20)
        )
    )

  save_fn <- paste("analysis_data/figures/time/", vec, ".png", sep = "")

  ggsave(
    save_fn
  )
  print(paste("Saved an image to", save_fn))

}


for (vec in unique(df_analysis$vectorizer)) {
    df_vec <- df_analysis %>% filter(
        vectorizer == vec
    ) %>% mutate(
    year_med_z = scale(year_med),
    ref_med_z = scale(ref_med),
  ) %>% filter(
    ref_med_z < 3,
    ref_med_z > -3,
    )

    (
        ggplot(
            df_vec,
            aes(
                x=density_bin_z,
                y=ref_med_z,
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
            alpha = 0.2, 
        )
        + geom_smooth(
            aes(
                color = field,
            ),
            linewidth = 1,
        )
        + theme_classic()
        + ggtitle(vec)
        + theme(
            axis.text = element_text(size=20)
        )
    )

  save_fn <- paste("analysis_data/figures/refs/", vec, ".png", sep = "")

  ggsave(
    save_fn
  )
  print(paste("Saved an image to", save_fn))

}

