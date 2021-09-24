import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def trim_method_str(method_str: str):
    method_list = method_str.split("/")
    method = method_list[0]
    if len(method_list) == 1:
        return method
    model = method_list[1]
    if model.__contains__("VGG"):
        model = "VGG"
    elif model.__contains__("Resnet"):
        model = "Resnet"
    return method + "/" + model


def save_results_as_csv(results, filename="results.csv"):
    df = pd.DataFrame(results, columns=["m", "ds", "dv", "dt", "method", "a", "acc"])
    df = df.drop(["m", "a"], axis=1)
    df.method = df.method.apply(lambda x: trim_method_str(x))
    df = df[df.method != 'MCDropout-7/MNIST_Simple']
    df.to_csv(filename)


def show_values_on_bars(axs):
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.3f}'.format(p.get_height())
            ax.text(_x, _y + 0.05, value, ha="center")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


def draw(filename="results.csv"):
    df = pd.read_csv(filename, index_col=0)

    print(df)
    plot_order = df.groupby(["method"])['acc'].mean().sort_values()
    print(plot_order.index)
    print(type(plot_order))
    g = sns.catplot(
        data=df, kind="bar",
        x="method", y="acc", palette="dark", alpha=.6, order=plot_order.index
    )
    g.despine(left=True)
    g.set_axis_labels("", "Accuracy")
    show_values_on_bars(g.ax)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    # plt.savefig("wykres.png")


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
    # draw_boxplots()
    results = pd.read_csv("results/results_working_methods.csv", index_col=0)
    results2 = pd.read_csv("results/results_fixed_methods.csv", index_col=0)
    results3 = pd.read_csv("results/results_resnet_all_datasets.csv", index_col=0)
    results4 = pd.read_csv("results/results_vgg_all_datasets.csv", index_col=0)
    save_results_as_csv(pd.concat([results, results2, results3, results4]), "results/results_all.csv")
    draw("results/results_all.csv")