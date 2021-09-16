from utils.plot import save_results_as_csv
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def draw_boxplots():

    df_valid = pd.read_csv("results/valid.csv")
    df_test = pd.read_csv("results/test.csv")
    # incorrect = df_test[~df_test["correct"]]


    concat = pd.concat([df_test, df_valid],
                       keys=["test_set", "cifar"], names=["dataset_type", "num"])
    concat.reset_index(inplace=True)
    _ = sns.catplot(x="dataset_type", y="comfort_level", kind="box", data=concat)
    plt.show()

if __name__ == "__main__":
    draw_boxplots()
    # results = pd.read_csv("results/results2.csv", index_col=0)
    # results2 = pd.read_csv("results/results3.csv", index_col=0)
    # save_results_as_csv(pd.concat([results, results2]), "results/results22.csv")