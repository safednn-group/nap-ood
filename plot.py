import os.path
import glob
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
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y + 0.01, value, ha="center", rotation=90)

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


def draw_hamming_distances():
    d1_tasks = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet']
    for d1 in d1_tasks:
        train_file_pattern = "*distances2*_" + d1 + "_*.csv"
        test_file_pattern = "*testdistances*_" + d1 + "_vs*.csv"
        train_files = glob.glob(os.path.join("results/distances", train_file_pattern))
        test_files = glob.glob(os.path.join("results/distances", test_file_pattern))

        li = []
        model = None
        dataset = None
        test_dataset = None
        for filename in train_files:
            df = pd.read_csv(filename, index_col=0)
            # print(df)
            # df.plot.hist(bins=df["hamming_distance"].max())
            split = filename.split(".")[0].split("_")[1:]
            title = "".join(split)
            model = split[0]
            dataset = split[2]
            # plt.title(title)
            # # # plt.show()
            # plt.savefig(os.path.join("results/distances/plots", title))
            # plt.close()
            li.append(df)

        frame = pd.concat(li, axis=0, ignore_index=True)
        print(len(frame.index))
        print(len(frame.sample(10)))
        for filename in test_files:
            df = pd.read_csv(filename, index_col=0)
            frame_train_sampled = frame.sample(len(df.index))
            print(frame_train_sampled["hamming_distance"].mean())
            print(len(frame_train_sampled.index))
            print(len(df.index))
            split = filename.split(".")[0].split("_")[1:]
            title = "".join(split)
            title = "compared" + title
            plt.figure()
            _ = plt.hist(frame_train_sampled, bins=int(df["hamming_distance"].max() / 4), alpha=0.7, label='train')
            _ = plt.hist(df, bins=int(df["hamming_distance"].max() / 4), alpha=0.7, label='test')
            plt.legend(loc='lower right')
            threshold, acc = find_threshold(frame_train_sampled, df)
            plt.axvline(threshold, color='k', linestyle='dashed', linewidth=1)
            min_ylim, max_ylim = plt.ylim()
            plt.text(threshold * 1.1, max_ylim * 0.9, f'Threshold: {threshold}, Accuracy: {acc}')
            plt.title(title)
            plt.savefig(os.path.join("results/distances/plots", title))
            plt.close()

    # frame.plot.hist(bins=df["hamming_distance"].max(), alpha=0.5, label="train")
    # title = "merged2" + model + dataset
    # plt.savefig(os.path.join("results/distances/plots", title))
    # plt.show()
    # print(frame)


def fix_vgg_results():
    all_files = glob.glob(os.path.join("results/article_plots", "VGG*"))
    wrong_layers = {7, 8, 13, 14}
    for filename in all_files:
        df = pd.read_csv(filename, index_col=0)
        for i in wrong_layers:
            ds = df.loc[df["layer"] == i][["ds"]].values
            dt = df.loc[df["layer"] == i][["dt"]].values
            if ds.size:
                ds = ds[0][0]
            if dt.size:
                dt = dt[0][0]
            df.loc[df["layer"] == i, "ds"] = dt
            df.loc[df["layer"] == i, "dt"] = ds
        df.to_csv(filename)


def find_threshold(df_known, df_unknown):
    min = df_unknown["hamming_distance"].min() if df_unknown["hamming_distance"].min() > df_known[
        "hamming_distance"].min() else \
        df_known["hamming_distance"].min()
    max = df_unknown["hamming_distance"].max() if df_unknown["hamming_distance"].max() > df_known[
        "hamming_distance"].max() else \
        df_known["hamming_distance"].max()
    best_correct_count = 0
    best_threshold = 0
    for i in range(min - 1, max + 1):
        correct_count = 0
        correct_count += (df_unknown["hamming_distance"] > i).sum()
        correct_count += (df_known["hamming_distance"] <= i).sum()
        if best_correct_count < correct_count:
            best_correct_count = correct_count
            best_threshold = i
    print(f" best threshold: {best_threshold}")
    print(f" accuracy: {best_correct_count / (len(df_unknown.index) + len(df_known.index))}")
    return best_threshold, best_correct_count / (len(df_unknown.index) + len(df_known.index))


def draw_article_plots():
    all_files = glob.glob(os.path.join("results/article_plots", "Re*"))

    li = []
    model = None
    dataset = None
    for filename in all_files:
        df = pd.read_csv(filename, index_col=0)
        li.append(df)

    print(len(li))
    frame = pd.concat(li, axis=0, ignore_index=True)
    print(frame)
    # grouped = frame.groupby(["quantile", "layer"])["test_acc"].mean().sort_values()
    # grouped = frame.groupby(["quantile", "pool"])["test_acc"].mean().sort_values()
    grouped = frame.groupby(["quantile"])["test_acc"].mean().sort_values()
    g = grouped.plot(kind="bar")
    show_values_on_bars(g)
    # quantile_formatter = lambda x: '(' + '{:.2f}'.format(x[0]) + ', ' + str(x[1]) + ', ' + str(x[2]) + ')'
    quantile_formatter = lambda x: '(' + '{:.2f}'.format(x[0]) + ', ' + str(x[1]) + ')'
    # g.set_xticklabels([quantile_formatter(eval(x.get_text())) for x in g.get_xticklabels()])
    title = "Resnet_CIFAR100_quantile"
    plt.title(title)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join("results/article_plots/plots", title))


if __name__ == "__main__":
    # draw_boxplots()
    # results = pd.read_csv("results/results_working_methods.csv", index_col=0)
    # results2 = pd.read_csv("results/results_fixed_methods.csv", index_col=0)
    # results3 = pd.read_csv("results/results_resnet_all_datasets.csv", index_col=0)
    # results4 = pd.read_csv("results/results_vgg_all_datasets.csv", index_col=0)
    # save_results_as_csv(pd.concat([results, results2, results3, results4]), "results/results_all.csv")
    # draw("results/results_all.csv")
    # draw_hamming_distances()
    draw_article_plots()
    # fix_vgg_results()
