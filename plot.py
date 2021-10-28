import os.path
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
from pylatex import Document, Section, Figure, NoEscape, Subsection, NewPage
from PIL import Image

d2_compatiblity = {
    # This can be used as d2 for            # this
    'MNIST': ["UniformNoise", "NormalNoise", 'FashionMNIST', 'NotMNIST',  'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet'],
    'FashionMNIST': ["UniformNoise", "NormalNoise", 'MNIST', 'NotMNIST',  'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet'],
    'CIFAR10': ["UniformNoise", "NormalNoise", 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR100', 'TinyImagenet'],
    'CIFAR100': ["UniformNoise", "NormalNoise", 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'STL10', 'TinyImagenet'],
    'STL10': ["UniformNoise", "NormalNoise", 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR100', 'TinyImagenet'],
    'TinyImagenet': ["UniformNoise", "NormalNoise", 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'CIFAR100', 'STL10'],
    # STL10 is not compatible with CIFAR10 because of the 9-overlapping classes.
    # Erring on the side of caution.
}


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
    lengths = {
        "MNIST": 128,
        "FashionMNIST": 2048,
        "CIFAR10": 4096,
        "CIFAR100": 2048,
        "STL10": 2048,
        "TinyImagenet": 512,
    }
    quantiles = {
        "MNIST": 0.5,
        "FashionMNIST": 0.9,
        "CIFAR10": 0.9,
        "CIFAR100": 0.9,
        "STL10": 0.9,
        "TinyImagenet": 0.5,
    }
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
            plt.xlabel(f'Vector length: {lengths[d1]}, {(1 - quantiles[d1]):.1f}% highest values considered')
            plt.title(title)
            # plt.show()
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


def draw_article_plots(ds, dv=None):
    plt.close()
    all_files = glob.glob(os.path.join("results/article_plots", ("VGG" + ds + "*")))
    li = []
    for filename in all_files:
        if ds == "CIFAR10":
            if filename.split('/')[2][10] != '0':
                df = pd.read_csv(filename, index_col=0)
                li.append(df)
        else:
            df = pd.read_csv(filename, index_col=0)
            li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    if dv:
        grouped = frame.groupby(["quantile", "layer", "pool", "dv"])[
            "valid_acc", "test_acc", "threshold"].mean().sort_values("valid_acc")
        grouped = grouped.xs(dv, level="dv").tail(20)
        title = "VGG_" + ds + "_vs_" + dv
    else:
        grouped = frame.groupby(["quantile", "layer", "pool"])[
            "valid_acc", "test_acc", "threshold"].mean().sort_values("valid_acc").tail(20)
        title = "VGG_" + ds
    figure, axes = plt.subplots()
    x = np.arange(len(grouped.index))
    v = axes.bar(x - 0.2, grouped["valid_acc"].values, 0.4)
    t = axes.bar(x + 0.2, grouped["test_acc"].values, 0.4)
    show_values_on_bars(axes)
    quantile_formatter = lambda x: '(' + '{:.2f}'.format(x[0]) + ', ' + str(x[1]) + ', ' + str(x[2]) + ')'
    # quantile_formatter = lambda x: '(' + '{:.2f}'.format(x[0]) + ', ' + str(x[1]) + ')'
    # quantile_formatter = lambda x: '(' + '{:.2f}'.format(x) + ')'
    plt.xticks(x, grouped.index, rotation=90)
    axes.set_xticklabels([quantile_formatter(eval(x.get_text())) for x in axes.get_xticklabels()])

    plt.title(title)

    plt.tight_layout()
    # plt.show()
    # plt.savefig(os.path.join("results/article_plots/plots", title))

def load_distance(filename):
    plt.close()
    img = Image.open(filename)
    plt.axis("off")
    plt.imshow(img)
    split = filename.split(".")[0].split("model")[1].split("dataset")
    title = "".join(split)
    model = split[0]
    dataset = split[1].split("vs")[0]
    valid_dataset = split[1].split("vs")[1]
    plt.title(title)

def generate_latex(fname, width, *args, **kwargs):
    d1_tasks = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet']

    # d1_tasks = ['MNIST', 'CIFAR100', 'STL10']
    d2_tasks = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'CIFAR100', 'STL10',
                'TinyImagenet']
    geometry_options = {"right": "2cm", "left": "2cm"}
    doc = Document(fname, geometry_options=geometry_options)
    with doc.create(Section("Opis")):
        doc.append("Na wykresach w sekcji nr. 2 widać, że występuje zależność pomiędzy skutecznością danej konfiguracji na zbiorze walidacyjnym i testowym. \n")
        doc.append("Można na tej informacji oprzeć metodę adaptacyjną, bo zbiorem walidacyjnym dysponujemy w trakcie trenowania metody. I w większości przypadków byłoby to skuteczne podejście.\n")
        doc.append("Niestety na wykresach w sekcji nr. 3, w szczególności na wykresie VGG_CIFAR100vsMNIST pokazana jest pułapka tego podejścia. \n")
        doc.append("Czasami słabe w ogólności konfiguracje pozwalają znaleźć najlepszy próg oddzielający zbiór treningowy od zupełnie różnego zbioru walidacyjnego takiego jak NormalNoise. \n")
        doc.append("Pomysły mam dwa \n")
        doc.append("1. Algorytm treningu i działania metody byłby następujący:  \n")
        doc.append("\t -  Dla każdej konfiguracji z hipersiatki możliwych konfiguracji wyznacz threshold i skuteczność vs dataset walidacyjny\n")
        doc.append("\t  - Dla n najlepszych warstw wybierz najlepszą konfigurację dla danej warstwy i zapamiętaj n zbiorów patternów treningowych (jeden dla każdej konfiguracji) i threshold dla niej wyznaczony\n")
        doc.append("Faza testowa: \n")
        doc.append("\t  - Wyznacz n patternów dla próbki testowej (jeden dla każdej konfiguracji; wszystkie n podczas jedenego forward passu przez sieć rzecz jasna)\n")
        doc.append("\t  - Wyznacz dystans n patternów z odpowiadającymi zbiorami patternów znanych i porównaj z thresholdami\n")
        doc.append("\t  - Głosowanie warstw - jeśli i-ty pattern porównany z i-tym zbiorem znanych patternów jest poniżej thresholdu to warstwa głosuje, że pattern jest znany\n")
        doc.append(NewPage())
        doc.append("Liczba n - branych pod uwagę warstw powinna być nieparzysta, albo trzeba by zastosować ważenie głosów (lepsza skuteczność na zbiorze walidacyjnym -> większa waga) \n")
        doc.append("Moim zdaniem ma to potencjał dać dobre wyniki - dla takich przypadków jak CIFAR100vsNoise/MNIST da na pewno gorszy wynik, niż najlepsza ogólnie pojedyncza warstwa, lecz nie mamy dostępu do wyroczni.\n")
        doc.append("A dla pozostałych przypadków niewykluczone, że otrzymamy nawet lepsze wyniki dzięki fazie głosowania\n")
        doc.append("\n 2. By zniwelować niekorzystny efekt opisany wyżej przy uczeniu metody przeciw takim datasetom jak NormalNoise można spróbować podczas treningu wykorzystać substytut podobnego datasetu w postaci próbek należących do części klas znanego datasetu.\n")
        doc.append(" Wyglądałoby to mniej więcej tak: \n")
        doc.append(" - wybieramy n klas zbioru treningowego np. 30%\n")
        doc.append(" - szukamy najlepszej konfiguracji podobnie jak wcześniej, tylko wybór opieramy jeszcze na informacji, która warstwa najlepiej separowała znany zbiór od sztucznie wygenerowanego podobnego nieznanego zbioru \n")
        doc.append(" - załóżmy, że zbiorem treningowym jest CIFAR100 \n")
        doc.append(" - wybieramy klasy 80-100 jako nieznane i tworzymy z nich zbiór walidacyjny; zapamiętujemy patterny pozostałych 80 klas\n")
        doc.append(" - gdy sieć sklasyfikuje jakąś próbkę jako jedną z klas 80-100, losowo wybieramy, od której klasy zapamiętanych patternów (0-79) liczymy odległość Hamminga\n")
        doc.append("Nie jestem w stanie przewidzieć, czy ten sposób by coś nam dał, ale może udałoby się dzięki temu wybierać lepsze konfiguracje.")
        doc.append(NewPage())

    with doc.create(Section('Wykresy konfiguracji zagregowane po wszystkich datasetach walidacyjnych (20 najlepszych wyników)')):
        for d1 in d1_tasks:
            plot = Figure(position='h')
            draw_article_plots(d1)
            plot.add_plot(width=NoEscape(width), *args, **kwargs)
            plot.add_caption('X - konfiguracje; Y - valid_accuracy (niebieski), test_accuracy (pomaranczowy).')
            doc.append(plot)
            doc.append(NewPage())

    with doc.create(Section('Wykresy konfiguracji (20 najlepszych wyników)')):
        for d1 in d1_tasks:
            for d2 in d2_tasks:
                if d2 in d2_compatiblity[d1]:
                    plot = Figure(position='h')
                    draw_article_plots(d1, d2)
                    plot.add_plot(width=NoEscape(width), *args, **kwargs)
                    plot.add_caption('X - konfiguracje; Y - valid_accuracy (niebieski), test_accuracy (pomaranczowy).')
                    doc.append(plot)
                    doc.append(NewPage())

    with doc.create(Section('Wykresy dystansów wzorcow aktywacji')):
        all_files = glob.glob(os.path.join("results/distances/plots", "compared*"))
        for filename in sorted(all_files):
            plot = Figure(position='h')
            load_distance(filename)
            plot.add_plot(width=NoEscape(width), *args, **kwargs)
            plot.add_caption('X - dystans; Y - liczba powtorzen.')
            doc.append(plot)
            doc.append(NewPage())

    doc.generate_pdf(clean_tex=False)


def draw_activations(patterns, shapes):
    ppc_h, ppc_w = 20, 60
    rows = np.min(shapes)
    ax_labels = []
    layers_dict = {}
    shapes.reverse()
    for i, shape in enumerate(shapes):
        layer_label_len = int(math.ceil(shape/rows))
        layer_cols = []
        for j in range(layer_label_len):
            label = str(i) + "." + str(j)
            ax_labels.append(label)
            layer_cols.append(label)
        layers_dict[i] = layer_cols
    cols = len(ax_labels)
    img_h, img_w = rows * ppc_h, cols * ppc_w
    neuron_h, neuron_w = img_h // rows, img_w // cols
    for pattern in range(patterns.shape[0]):
        img = np.ones((img_h, img_w, 3))
        fig, ax = plt.subplots(1)
        fig.set_size_inches(18.5, 10.5, forward=True)
        ax.imshow(img)
        offset = 0
        col_offset = 0
        for j, shape in enumerate(shapes):
            for col_id, col in enumerate(layers_dict[j]):
                for i in range(rows):
                    if col_id * rows + i > shape:
                        break
                    if patterns[pattern, offset].item() > 0:
                        rect = patches.Rectangle((neuron_w * col_offset, neuron_h * i), neuron_w, neuron_h, linewidth=0,
                                                 edgecolor='black', facecolor="r")
                        ax.add_patch(rect)
                    else:
                        rect = patches.Rectangle((neuron_w * col_offset, neuron_h * i), neuron_w, neuron_h, linewidth=0,
                                                 edgecolor='black', facecolor="none")
                        ax.add_patch(rect)
                    offset += 1
                col_offset += 1
        plt.xticks(np.arange(len(ax_labels)) * ppc_w, ax_labels)
        plt.xlabel("layer_num.part")
        plt.ylabel("neuron_row_num")
        plt.yticks(np.arange(rows) * ppc_h, np.arange(rows))
        plt.show()

if __name__ == "__main__":
    # draw_boxplots()
    # results = pd.read_csv("results/results_working_methods.csv", index_col=0)
    # results2 = pd.read_csv("results/results_fixed_methods.csv", index_col=0)
    # results3 = pd.read_csv("results/results_resnet_all_datasets.csv", index_col=0)
    # results4 = pd.read_csv("results/results_vgg_all_datasets.csv", index_col=0)
    # save_results_as_csv(pd.concat([results, results2, results3, results4]), "results/results_all.csv")
    # draw("results/results_all.csv")
    # draw_hamming_distances()
    # draw_article_plots()
    generate_latex('matplotlib_ex-dpi', r'1\textwidth', dpi=100)
    # fix_vgg_results()
