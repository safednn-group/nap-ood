from utils.plot import save_results_as_csv
import torch
import pandas as pd

if __name__ == "__main__":
    results = pd.read_csv("results/results2.csv", index_col=0)
    results2 = pd.read_csv("results/results3.csv", index_col=0)
    save_results_as_csv(pd.concat([results, results2]), "results/results22.csv")