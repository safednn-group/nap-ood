import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def trim_method_str(method_str: str):
    method_list = method_str.split("/")
    method = method_list[0]
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
    df.to_csv(filename)

    draw(filename)


def draw(filename="results.csv"):
    df = pd.read_csv(filename, index_col=0)

    print(df)
    g = sns.catplot(
        data=df, kind="bar",
        x="method", y="acc",  palette="dark", alpha=.6
    )
    g.despine(left=True)
    g.set_axis_labels("", "Accuracy")
    plt.show()


