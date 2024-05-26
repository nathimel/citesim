"""Simple script to parse the .txt files containing results of regressions, and checks/reports the variance explained by field and density_bin, and the BIC difference between ablated model. """

import os
import pandas as pd
from typing import Callable

ANOVA_RESULTS_DIR = "./analysis_data/stats/anova_ablation"
MODEL_SUMMARY_DIR = "./analysis_data/stats/summary"

def tokenize(lines: list[str]) -> list[list[str]]:
    return [line.split() for line in lines]

def process_anova(data: list[list[str]]) -> tuple[float]:
    model_ablated_bic = None
    model_w_rho = None
    
    for tokens in data:
        if tokens[0] == "model_ablated":
            model_ablated_bic = tokens[3]
        elif tokens[0] == "model_w_rho":
            model_w_rho = tokens[3]

    return (float(model_ablated_bic) - float(model_w_rho),)

def process_summary(data: list[list[str]]) -> tuple[float]:

    for num, tokens in enumerate(data):
        if tokens[:2] == ["Random","effects:"]:
            # Parse the lines containing effects variance explained
            lines = data[num+2:num+5]
            field_var = float(lines[0][2])
            rho_var = float(lines[1][1])
            res_var = float(lines[2][1])

            total = field_var + rho_var + res_var
            field_var_perc = (field_var / total) * 100
            rho_var_perc = (rho_var / total) * 100
            res_var = (res_var / total) * 100
            break
    
    return (field_var_perc, rho_var_perc)

    
def process(mode: str) -> list[str]:

    if mode == "anova":
        dir = ANOVA_RESULTS_DIR
        process_func = process_anova
        cols = ["bic_gain"]
    elif mode == "summary":
        dir = MODEL_SUMMARY_DIR
        process_func = process_summary
        cols = ["field_var_percent", "density_var_percent"]

    data = []
    for filename in os.listdir(dir):
        # we now have three regressions so let's not mix them up
        vectorizer = filename.split("_")[-1].split(".txt")[0]

        with open(os.path.join(dir, filename), "r") as f:
            lines = f.readlines()

        tokens = tokenize(lines)
        data.append(
            (vectorizer, *process_func(tokens))
        )
    stat_df = pd.DataFrame(data, columns=["vectorizer"] + cols)
    return stat_df


if __name__ == "__main__":

    # Anova
    df = process("anova")
    print("---------------ANOVA---------------")
    print(df)
    print()

    # Summary
    df = process("summary")
    print("---------------SUMMARY---------------")
    print(df)
    print()
    