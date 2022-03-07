import os.path
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math


from numpy.ma.core import choose
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import torch
from pylatex import Document, Section, Figure, NoEscape, Subsection, NewPage, SubFigure, PageStyle
from PIL import Image

d2_compatiblity = {
    # This can be used as d2 for            # this
    'MNIST': ["UniformNoise", "NormalNoise", 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'CIFAR100', 'STL10',
              'TinyImagenet'],
    'FashionMNIST': ["UniformNoise", "NormalNoise", 'MNIST', 'NotMNIST', 'CIFAR10', 'CIFAR100', 'STL10',
                     'TinyImagenet'],
    'CIFAR10': ["UniformNoise", "NormalNoise", 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR100', 'TinyImagenet'],
    'CIFAR100': ["UniformNoise", "NormalNoise", 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'STL10',
                 'TinyImagenet'],
    'STL10': ["UniformNoise", "NormalNoise", 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR100', 'TinyImagenet'],
    'TinyImagenet': ["UniformNoise", "NormalNoise", 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'CIFAR100',
                     'STL10'],
    # STL10 is not compatible with CIFAR10 because of the 9-overlapping classes.
    # Erring on the side of caution.
}

layers_shapes = {
    "VGG": {
        'MNIST': [256, 256, 512, 512, 256, 256, 128, 128, 64],
        'FashionMNIST': [256, 256, 512, 512, 256, 256, 128, 128, 64],
        'CIFAR10': [4096, 4096, 512, 512, 512, 512, 512, 512, 256, 256, 256, 128, 128, 64, 64],
        'CIFAR100': [4096, 4096, 512, 512, 512, 512, 512, 512, 256, 256, 256, 128, 128, 64, 64],
        'STL10': [4096, 4096, 512, 512, 512, 512, 512, 512, 256, 256, 256, 128, 128, 64, 64],
        'TinyImagenet': [4096, 4096, 512, 512, 512, 512, 512, 512, 256, 256, 256, 128, 128, 64, 64],
    },
    "Resnet": {
        'MNIST': [2048, 2048, 1024, 1024, 1024, 1024, 1024, 512, 512, 512, 256, 256],
        'FashionMNIST': [2048, 2048, 1024, 1024, 1024, 1024, 1024, 512, 512, 512, 256, 256],
        'CIFAR10': [2048, 2048, 2048, 1024, 1024, 1024, 1024, 1024, 1024, 512, 512, 512, 512, 256, 256, 256],
        'CIFAR100': [2048, 2048, 2048, 1024, 1024, 1024, 1024, 1024, 1024, 512, 512, 512, 512, 256, 256, 256],
        'STL10': [2048, 2048, 2048, 1024, 1024, 1024, 1024, 1024, 1024, 512, 512, 512, 512, 256, 256, 256],
        'TinyImagenet': [2048, 2048, 2048, 1024, 1024, 1024, 1024, 1024, 1024, 512, 512, 512, 512, 256, 256, 256],
    }

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
    df = pd.DataFrame(results, columns=["m", "ds", "dv", "dt", "method", "a", "acc", "auroc", "aupr"])
    df = df.drop(["m", "a"], axis=1)
    df.method = df.method.apply(lambda x: trim_method_str(x))
    df = df[df.method != 'MCDropout-7/MNIST_Simple']
    df.to_csv(filename)


def draw(filename="results.csv", metric="acc", label="Mean test accuracy"):
    df = pd.read_csv(filename, index_col=0)
    # df = df[df["ds"] != "TinyImagenet"]
    # df = df[df["ds"] == "STL10"]
    print(df)
    plot_order = df.groupby(["method"])[metric].mean().sort_values()
    print(plot_order.index)
    print(type(plot_order))
    g = sns.catplot(
        data=df, kind="bar",
        x="method", y=metric, palette="dark", alpha=.6, order=plot_order.index
    )
    g.despine(left=True)
    g.set_axis_labels("Metoda", label)
    show_values_on_bars(g.ax)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    # plt.savefig(metric + filename.split(".")[0] + ".png")


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


def draw_hamming_distances_layerwise():
    d1_tasks = ["FashionMNIST"]
    nap_cfg_path = "nap_cfgs/full_nets.json"
    import json
    with open(nap_cfg_path) as cf:
        nap_cfg = json.load(cf)
    for d1 in d1_tasks:
        for model in ["VGG"]:
            for layer in range(len(layers_shapes[model][d1])):
                train_file_pattern = "q_0.3_traindistances_model" + model + "_dataset_" + d1 + "_" + str(layer) + ".csv"
                test_file_pattern = "q_0.3_testdistances_model" + model + "_dataset_" + d1 + "*" + str(layer) + ".csv"
                train_files = glob.glob(os.path.join("results/distances/redo", train_file_pattern))
                test_files = glob.glob(os.path.join("results/distances/redo", test_file_pattern))
                frame = pd.read_csv(train_files[0], index_col=0)

                print(len(frame.index))
                print(len(frame.sample(10)))
                for filename in test_files:
                    df = pd.read_csv(filename, index_col=0)
                    if len(df.index) > len(frame.index):
                        print(filename)
                        print(len(df.index))
                        print(train_files[0])
                        print(len(frame.index))
                        frame_train_sampled = frame
                        df = df.sample(len(frame.index))
                    else:
                        frame_train_sampled = frame.sample(len(df.index))
                    print(frame_train_sampled["hamming_distance"].mean())
                    print(len(frame_train_sampled.index))
                    print(len(df.index))
                    split = filename.split("_")[1:]
                    title = "".join(split)
                    split = title.split(".")[:-1]
                    title = "".join(split)
                    title1 = "compared" + title
                    title = title1.split("dataset")[-1]
                    l = title[-1]
                    d2 = title[:-1].split("vs")[1]
                    title = title[:-1] + " layer no. " + l

                    plt.figure()
                    # _ = plt.hist(frame_train_sampled, bins=int(df["hamming_distance"].max() / 4) + 1, alpha=0.7, label='train')
                    # _ = plt.hist(df, bins=int(df["hamming_distance"].max() / 4) + 1, alpha=0.7, label='test')
                    m = max(df["hamming_distance"].max(), frame_train_sampled["hamming_distance"].max())
                    if df["hamming_distance"].max() < 20:
                        bins = int(m)
                    elif df["hamming_distance"].max() < 40:
                        bins = int(m / 2)
                    elif df["hamming_distance"].max() < 80:
                        bins = int(m / 3)
                    else:
                        bins = int(m / 4)

                    (_, bins1, _) = plt.hist(frame_train_sampled, bins=bins, alpha=0.7, label='znany rozkład')
                    (_, bins2, _) = plt.hist(df, bins=bins, alpha=0.7, label='nieznany rozkład')

                    plt.legend(loc='upper right')
                    threshold, acc = find_threshold(frame_train_sampled, df)
                    threshold += 1
                    plt.axvline(threshold, color='k', linestyle='dashed', linewidth=1)
                    min_ylim, max_ylim = plt.ylim()
                    plt.text(5, max_ylim * 0.7, f'Próg: {threshold}, celność: {acc:.3f}')
                    # plt.xlabel(f'Pattern length: {layers_shapes[model][d1][layer]}, p = { int(nap_cfg[model][d1][str(layer)]["quantile"] * 100)} (activation function percentile parameter)')
                    plt.ylabel(f'Liczba próbek')
                    # plt.xlabel(f'Hamming distance from the nearest known (training) pattern ')
                    plt.xlabel(f'Odległość Hamminga do najbliższej znanej próbki treningowej')
                    # plt.title(title)
                    # plt.show()
                    # plt.tight_layout()
                    plt.savefig(os.path.join("results/distances/plots/redo", title1))
                    plt.close()


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


def draw_article_plots(ds, dv=None, model="VGG", q=0.3):
    plt.close()
    all_files = glob.glob(os.path.join("results/article_plots", (model + ds + "*otherlayers.csv")))
    li = []
    nap_cfg_path = "nap_cfgs/full_nets.json"
    import json
    with open(nap_cfg_path) as cf:
        nap_cfg = json.load(cf)
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
        frame = frame[frame["pool_type"] == nap_cfg[model][ds]["1"]["pool_type"]]
        # frame = frame[frame["quantile"] != 0.1]
        # frame = frame[frame["quantile"] != 0.5]
        # frame = frame[~np.isclose(frame["quantile"],  0.7)]
        frame = frame[frame["quantile"] == q]

        grouped = frame.groupby(["quantile", "layer", "pool_type", "dv"])[
            "valid_acc", "test_acc", "threshold"].mean().sort_values("valid_acc")
        grouped = grouped.xs(dv, level="dv")

        title = "VGG_" + ds + "_vs_" + dv
    else:
        grouped = frame.groupby(["quantile", "layer", "pool_type"])[
            "valid_acc", "test_acc", "threshold"].mean().sort_values("valid_acc")
        title = "VGG_" + ds
    figure, axes = plt.subplots()
    x = np.arange(len(grouped.index))
    v = axes.bar(x - 0.2, grouped["valid_acc"].values, 0.4, label="cel. walid.")
    t = axes.bar(x + 0.2, grouped["test_acc"].values, 0.4, label="śr. cel. test.")

    show_values_on_bars(axes)
    # quantile_formatter = lambda x: '(' + 'p - {}'.format(int(x[0] * 100)) + ', ' + str(x[1]) + ', ' + str(x[2]) + ')'
    quantile_formatter = lambda x: 'Layer: ' + str(x[1])
    # quantile_formatter = lambda x: '(' + '{:.2f}'.format(x) + ')'
    plt.xticks(x, grouped.index, rotation=90)

    axes.set_xticklabels([quantile_formatter(eval(x.get_text())) for x in axes.get_xticklabels()])

    # plt.title(title)
    # plt.xticks([])
    plt.xlabel("Konfiguracje NAP'owe")
    plt.ylabel("Uśredniona celność")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join("results/article_plots/plots", title))


def load_distance(filename):
    plt.close()
    img = Image.open(filename)
    plt.axis("off")
    plt.imshow(img)
    split = filename.split(".")[0].split("model")[1].split("dataset")
    title = "".join(split)
    plt.title(title)


def generate_latex(fname, width, *args, **kwargs):
    d1_tasks = ['MNIST', 'FashionMNIST', 'CIFAR10', 'STL10', 'CIFAR100', 'TinyImagenet']

    d1_tasks = ['FashionMNIST']
    d2_tasks = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'STL10', 'CIFAR100',
                'TinyImagenet']
    geometry_options = {"right": "2cm", "left": "2cm"}
    doc = Document(fname, geometry_options=geometry_options)
    with doc.create(Section("Opis")):
        doc.append(
            "Na wykresach w sekcji nr. 2 widać, że występuje zależność pomiędzy skutecznością danej konfiguracji na zbiorze walidacyjnym i testowym. \n")
        doc.append(
            "Można na tej informacji oprzeć metodę adaptacyjną, bo zbiorem walidacyjnym dysponujemy w trakcie trenowania metody. I w większości przypadków byłoby to skuteczne podejście.\n")
        doc.append(
            "Niestety na wykresach w sekcji nr. 3, w szczególności na wykresie VGG_CIFAR100vsMNIST pokazana jest pułapka tego podejścia. \n")
        doc.append(
            "Czasami słabe w ogólności konfiguracje pozwalają znaleźć najlepszy próg oddzielający zbiór treningowy od zupełnie różnego zbioru walidacyjnego takiego jak NormalNoise. \n")
        doc.append("Pomysły mam dwa \n")
        doc.append("1. Algorytm treningu i działania metody byłby następujący:  \n")
        doc.append(
            "\t -  Dla każdej konfiguracji z hipersiatki możliwych konfiguracji wyznacz threshold i skuteczność vs dataset walidacyjny\n")
        doc.append(
            "\t  - Dla n najlepszych warstw wybierz najlepszą konfigurację dla danej warstwy i zapamiętaj n zbiorów patternów treningowych (jeden dla każdej konfiguracji) i threshold dla niej wyznaczony\n")
        doc.append("Faza testowa: \n")
        doc.append(
            "\t  - Wyznacz n patternów dla próbki testowej (jeden dla każdej konfiguracji; wszystkie n podczas jedenego forward passu przez sieć rzecz jasna)\n")
        doc.append(
            "\t  - Wyznacz dystans n patternów z odpowiadającymi zbiorami patternów znanych i porównaj z thresholdami\n")
        doc.append(
            "\t  - Głosowanie warstw - jeśli i-ty pattern porównany z i-tym zbiorem znanych patternów jest poniżej thresholdu to warstwa głosuje, że pattern jest znany\n")
        doc.append(NewPage())
        doc.append(
            "Liczba n - branych pod uwagę warstw powinna być nieparzysta, albo trzeba by zastosować ważenie głosów (lepsza skuteczność na zbiorze walidacyjnym -> większa waga) \n")
        doc.append(
            "Moim zdaniem ma to potencjał dać dobre wyniki - dla takich przypadków jak CIFAR100vsNoise/MNIST da na pewno gorszy wynik, niż najlepsza ogólnie pojedyncza warstwa, lecz nie mamy dostępu do wyroczni.\n")
        doc.append(
            "A dla pozostałych przypadków niewykluczone, że otrzymamy nawet lepsze wyniki dzięki fazie głosowania\n")
        doc.append(
            "\n 2. By zniwelować niekorzystny efekt opisany wyżej przy uczeniu metody przeciw takim datasetom jak NormalNoise można spróbować podczas treningu wykorzystać substytut podobnego datasetu w postaci próbek należących do części klas znanego datasetu.\n")
        doc.append(" Wyglądałoby to mniej więcej tak: \n")
        doc.append(" - wybieramy n klas zbioru treningowego np. 30%\n")
        doc.append(
            " - szukamy najlepszej konfiguracji podobnie jak wcześniej, tylko wybór opieramy jeszcze na informacji, która warstwa najlepiej separowała znany zbiór od sztucznie wygenerowanego podobnego nieznanego zbioru \n")
        doc.append(" - załóżmy, że zbiorem treningowym jest CIFAR100 \n")
        doc.append(
            " - wybieramy klasy 80-100 jako nieznane i tworzymy z nich zbiór walidacyjny; zapamiętujemy patterny pozostałych 80 klas\n")
        doc.append(
            " - gdy sieć sklasyfikuje jakąś próbkę jako jedną z klas 80-100, losowo wybieramy, od której klasy zapamiętanych patternów (0-79) liczymy odległość Hamminga\n")
        doc.append(
            "Nie jestem w stanie przewidzieć, czy ten sposób by coś nam dał, ale może udałoby się dzięki temu wybierać lepsze konfiguracje.")
        doc.append(NewPage())

    with doc.create(
            Section(
                'Wykresy konfiguracji zagregowane po wszystkich datasetach walidacyjnych (20 najlepszych wyników)')):
        for model in ["VGG"]:
            for d1 in d1_tasks:
                plot = Figure(position='h')
                draw_article_plots(d1, model=model)
                plot.add_plot(width=NoEscape(width), *args, **kwargs)
                plot.add_caption('X - konfiguracje; Y - valid_accuracy (niebieski), test_accuracy (pomaranczowy).')
                doc.append(plot)
                doc.append(NewPage())

    with doc.create(Section('Wykresy konfiguracji (20 najlepszych wyników)')):
        for model in ["VGG"]:
            for d1 in d1_tasks:
                for d2 in d2_tasks:
                    if d2 in d2_compatiblity[d1]:
                        plot = Figure(position='h')
                        draw_article_plots(d1, d2, model=model)
                        plot.add_plot(width=NoEscape(width), *args, **kwargs)
                        plot.add_caption(
                            'X - konfiguracje; Y - valid_accuracy (niebieski), test_accuracy (pomaranczowy).')
                        doc.append(plot)
                        doc.append(NewPage())

    # with doc.create(Section('Wykresy dystansów wzorcow aktywacji')):
    #     all_files = glob.glob(os.path.join("results/distances/plots/redo", "compared*datasetFashionMNIST*"))
    #     print("a")
    #     print(len(all_files))
    #     for filename in sorted(all_files):
    #         plot = Figure(position='h')
    #         load_distance(filename)
    #         plot.add_plot(width=NoEscape(width), *args, **kwargs)
    #         plot.add_caption('X - dystans; Y - liczba powtorzen.')
    #         doc.append(plot)
    #         doc.append(NewPage())

    doc.generate_pdf(clean_tex=False)


def generate_latex_heatmaps(fname, width, *args, **kwargs):
    d1_tasks = ['MNIST', 'FashionMNIST', 'CIFAR10', 'STL10', 'CIFAR100', 'TinyImagenet']

    d2_tasks = ['UniformNoise', 'NotMNIST', 'CIFAR10', 'TinyImagenet']
    geometry_options = {"head": "30pt",
                        "margin": "0.3in",
                        "top": "0.2in",
                        "bottom": "0.4in",
                        "includeheadfoot": True}

    doc = Document(fname, geometry_options=geometry_options)
    first_page = PageStyle("firstpage")
    doc.preamble.append(first_page)
    doc.change_document_style("firstpage")
    with doc.create(Section("Opis")):
        doc.append("Poniżej 3 rodzaje heatmap: \n")
        doc.append("\t 1. Zwykla heatmapa dla wszystkich patternów treningowych przykładowej klasy (0)  \n")
        doc.append("\t 2. Suma różnic pomiędzy heatmapą przykładowej klasy a każdą inną klasą \n")
        doc.append("\t 3. Różnica pomiędzy heatmapą dataset vs dataset \n")
        doc.append(
            "Ostatnie dwie warstwy VGG to \"classifier\", w przeciwieństwie do reszty warstw, są to warstwy Linear - nie są one poolowane - stąd widoczne różnice względem poprzednich warstw (więcej pól czarnych).\n")
        doc.append(
            "Dla większych VGG (nieMNIST'owych) wszystkie dwanaście warstw \"features\"  zajmuje jedynie 1/3 szerokości heatmapy. \n")
        doc.append(
            "Dla zobrazowania problemu: pierwsza warstwa \"features\" ma długość 64, a warstwy \"classifier\" mają po 4096.\n")
        doc.append("Liczność patternów składających się na heatmapę jest różna w zależności od datasetu \n")
        doc.append(
            "Brane pod uwagę są wartości od nastepujących centyli: MNIST, TinyImagenet - 50; FashionMNIST, CIFAR10, CIFAR100 - 90; STL10 - 37 \n")
        doc.append(
            "Generalnie widać, że heatmapy rodzaju 1. i 2. są do siebie podobne; właściwie dla każdego datasetu i każdej klasy istnieją takie neurony, które mają większe wartości aktywacji tylko dla jednej klasy (lub mniejsze, choć dla innych klas często przyjmują wartości większe). \n")
        doc.append(
            "Ta informacja potwierdza jedynie intuicję, że klasyfikacja jest pochodną konkretnych wzorców aktywacji.\n")
        doc.append(
            "Przy porównywaniu datasetów widać zależność - im większa amplituda różnicy heatmap tym łatwiej rozróżnialne są datasety widać to np. dla CIFAR100vsNoise i CIFAR100vsTinyImagenet\n")
        doc.append(
            "Ogólnie pozytywną wiadomością jest, że heatmapy są bardzo zróżnicowane pomiędzy klasami wewnątrz datasetu (bo klasyfikatory mają dużą dokładność),"
            " także pomiędzy większością datasetów wzajemnie, ale znowu pojawia się problem, że heatmapy różnic pomiędzy danym datasetem treningowym a dwoma istotnie różniącymi się datasetami walidacyjnymi także się między sobą różnią - np. MNISTvsNoise i MNISTvsNotMNIST \n")
        doc.append(NewPage())

    with doc.create(Section('Heatmapy')):
        for d1 in d1_tasks:
            plot = Figure(position='h')
            subsection = Subsection(d1 + " one class heatmap; same class minus the rest of classes")
            filename_class = "VGG_" + d1 + "_class0.png"
            filename_diffsum = "VGG_" + d1 + "_class0_diffsum.png"
            subfigure_class = SubFigure(position='c', width=NoEscape(r'0.45\linewidth'))
            subfigure_class.add_image(os.path.join("results/article_plots/heatmaps", filename_class),
                                      width=NoEscape(r'\linewidth'))
            subfigure_class.add_caption("VGG " + d1 + " class 0")
            subfigure_diffsum = SubFigure(position='c', width=NoEscape(r'0.45\linewidth'))
            subfigure_diffsum.add_image(os.path.join("results/article_plots/heatmaps", filename_diffsum),
                                        width=NoEscape(r'\linewidth'))
            subfigure_diffsum.add_caption("VGG " + d1 + " class 0 minus other classes")
            plot.append(subfigure_class)
            plot.append(subfigure_diffsum)
            doc.append(subsection)
            doc.append(plot)
            doc.append(NewPage())
            subsection = Subsection(d1 + " vs other datasets")
            doc.append(subsection)
            for d2 in d2_tasks:
                if d2 in d2_compatiblity[d1]:
                    subfigure_vs = Figure(position='h')
                    filename_vs = "VGG_" + d1 + "_vs_" + d2 + ".png"
                    subfigure_vs.add_image(os.path.join("results/article_plots/heatmaps", filename_vs),
                                           width=NoEscape(r'\linewidth'))
                    subfigure_vs.add_caption("VGG " + d1 + " vs " + d2)
                    doc.append(subfigure_vs)
                    doc.append(NewPage())

    doc.generate_pdf(clean_tex=False)


def draw_activations(patterns, shapes):
    ppc_h, ppc_w = 20, 60
    rows = np.min(shapes)
    ax_labels = []
    layers_dict = {}
    shapes.reverse()
    for i, shape in enumerate(shapes):
        layer_label_len = int(math.ceil(shape / rows))
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


def choose_layer(frames, thresholds, rownum, type=0, centile=1.):
    distances = []
    layernums = []
    for layernum in frames:
        df = frames[layernum]
        distance = df["distance"][rownum]
        if type == 0:
            distances.append(distance)
        elif type == 1:
            distances.append(abs(distance - thresholds["threshold"][int(layernum)]))
        elif type == 2:
            distances.append(thresholds["valid_acc"][int(layernum)])
        layernums.append(layernum)
    distances = np.array(distances)
    return layernums[distances.argsort()[int(len(distances) * centile) - 1]]
    # return layernums[distances.argmax()]


def choose_layers(frames, thresholds, rownum, type=0, votes=1):
    distances = []
    layernums = []
    for layernum in frames:
        df = frames[layernum]
        distance = df["distance"][rownum]
        if type == 0:
            distances.append(distance)
        elif type == 1:
            distances.append(abs(distance - thresholds["threshold"][int(layernum)]))
        elif type == 2:
            distances.append(thresholds["valid_acc"][int(layernum)])
        layernums.append(layernum)

    distances = np.array(distances)
    layernums = np.array(layernums)
    ids = distances.argsort()[::-1][:votes]
    # ids = np.append(distances.argsort()[::-1][:4], distances_.argsort()[::-1][:1])
    # if np.unique(ids.size) != 5:
    #     ids = np.append(ids, distances_.argsort()[::-1][1])
    # if np.unique(ids.size) != 5:
    #     ids = np.append(ids, distances.argsort()[::-1][4])
    # assert ids.size == 5
    return layernums[ids]


def linearize(frames, thresholds, dt, model):
    shapes = np.array(layers_shapes[model][dt])
    shape_factors = shapes / shapes.min()
    max_factor = shape_factors.max()
    thresholds_ = thresholds.copy()
    for layernum in frames:
        frames[layernum]["distance"] = frames[layernum]["distance"] + shape_factors[int(layernum)]
        frames[layernum]["distance"] = frames[layernum]["distance"] * (
                max_factor / shape_factors[int(layernum)])
        thresholds_["threshold"][int(layernum)] = thresholds_["threshold"][int(layernum)] + shape_factors[int(layernum)]
        thresholds_["threshold"][int(layernum)] = thresholds_["threshold"][int(layernum)] * (
                max_factor / shape_factors[int(layernum)])
    return thresholds_


def full_net_plot():
    d1_tasks = ['MNIST', 'FashionMNIST', 'CIFAR10', 'STL10', "TinyImagenet", "CIFAR100"]
    d2_tasks = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'STL10', 'CIFAR100',
                'TinyImagenet']
    d3_tasks = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'STL10', 'CIFAR100',
                'TinyImagenet']
    d1_tasks = ['MNIST', 'STL10', 'FashionMNIST', 'CIFAR100']
    # d2_tasks = ['NormalNoise']
    # d3_tasks = ['UniformNoise']
    types = [2]
    n_votes = [1]
    centiles = []

    for model_name in ["VGG", "Resnet"]:
        for type in types:
            for centile in centiles:
                agg_acc = 0
                counter = 0
                for d1 in d1_tasks:
                    for d2 in d2_tasks:
                        if d2 in d2_compatiblity[d1]:
                            df_thresholds = pd.read_csv(
                                "results/article_plots/full_nets/cut_tail/scoring/" + model_name + '_' + d1 + '_' + d2 + 'th-acc.csv',
                                index_col=0)

                            for d3 in d2_tasks:
                                if d2 != d3 and d3 in d2_compatiblity[d1]:
                                    file_pattern = model_name + '_' + d1 + '_' + d2 + '_' + d3 + "_*"
                                    files = glob.glob(
                                        os.path.join("results/article_plots/full_nets/cut_tail/scoring", file_pattern))
                                    frames = dict()
                                    rows = 0
                                    file_counter = 0
                                    for file in files:
                                        df = pd.read_csv(file, index_col=0)
                                        rows = len(df.index)
                                        layernum = file.split("_")[-1].split(".")[0]
                                        if df_thresholds["threshold"][int(layernum)] != -1:
                                            frames[layernum] = df
                                        # print(f"n: {file_counter}, {file}")
                                        file_counter += 1
                                    correct_count = 0
                                    thresholds_lin = linearize(frames, df_thresholds, d1, model_name)
                                    for i in range(rows):
                                        chosen_id = choose_layer(frames, thresholds_lin, i, type=type, centile=centile)
                                        correct_count += frames[chosen_id]["correct"][i]
                                    acc = correct_count / rows
                                    agg_acc += acc
                                    counter += 1
                    print(f"{d1} - type {type}, centile {centile} Aggregated accuracy: {agg_acc / counter}")

        for type in types:
            for votes in n_votes:
                agg_acc = 0
                counter = 0
                for d1 in d1_tasks:
                    votes = int(len(layers_shapes[model_name][d1]) / 3 + 1)
                    if votes % 2 == 0:
                        votes += 1
                    for d2 in d2_tasks:
                        if d2 in d2_compatiblity[d1]:
                            df_thresholds = pd.read_csv(
                                "results/article_plots/full_nets/cut_tail/scoring/" + model_name + '_' + d1 + '_' + d2 + 'th-acc.csv',
                                index_col=0)

                            for d3 in d3_tasks:
                                if d2 != d3 and d3 in d2_compatiblity[d1]:
                                    file_pattern = model_name + '_' + d1 + '_' + d2 + '_' + d3 + "_*"
                                    files = glob.glob(
                                        os.path.join("results/article_plots/full_nets/cut_tail/scoring", file_pattern))
                                    frames = dict()
                                    rows = 0
                                    for file in files:
                                        df = pd.read_csv(file, index_col=0)
                                        rows = len(df.index)
                                        layernum = file.split("_")[-1].split(".")[0]
                                        if df_thresholds["threshold"][int(layernum)] != -1:
                                            frames[layernum] = df
                                    correct_count = 0
                                    thresholds_lin = linearize(frames, df_thresholds, d1, model_name)
                                    # chosen_ids = dict()
                                    for i in range(rows):
                                        chosen_ids = choose_layers(frames, thresholds_lin, i, type=type, votes=votes)
                                        correct_votes = 0
                                        for chosen in chosen_ids:
                                            correct_votes += frames[chosen]["correct"][i]
                                        correct_count += (correct_votes > (len(chosen_ids) / 2))
                                        # if chosen_ids.get(chosen_id) is None:
                                        #     chosen_ids[chosen_id] = 0
                                        # else:
                                        #     chosen_ids[chosen_id] += 1

                                    acc = correct_count / rows
                                    # print(model_name + '_' + d1 + '_' + d2 + '_' + d3 + " acc: " + str(acc))
                                    agg_acc += acc
                                    counter += 1
                    print(f"{model_name}  {d1} - type {type}, votes {votes} Aggregated accuracy: {agg_acc / counter}")


def execution_times_plot():
    arr = np.load("results/article_plots/execution_times/servercifar10.npz")
    print(arr["avg_net_pass"])
    print(arr["avg_nap_net_pass"])
    print(arr["avg_compute_hamming"])
    print(arr["avg_compute_hamming_and"])
    print(arr["avg_compute_hamming_full_net"])
    print(arr["avg_compute_hamming_and_full_net"])
    x = np.arange(100, 4001, 300)[::-1]
    figure, axes = plt.subplots()
    _ = axes.plot(x, arr["avg_compute_hamming"], label="hamming_distance (xor) len=4096")
    _ = axes.plot(x, arr["avg_compute_hamming_and"], label="(xor) & (and) len=4096")
    _ = axes.plot(x, arr["avg_compute_hamming_full_net"], label="full net xor len=12416")
    _ = axes.plot(x, arr["avg_compute_hamming_and_full_net"], label="full net xor & and len=12416")
    plt.axhline(y=arr["avg_net_pass"], linestyle='-')
    min_xlim, max_xlim = plt.xlim()
    plt.text((max_xlim - min_xlim) / 2, arr["avg_net_pass"] * 1.1, f'Avg single net pass: {arr["avg_net_pass"]:.4f}')
    plt.text((max_xlim - min_xlim) / 2, arr["avg_nap_net_pass"] * 1.1,
             f'Avg single NAP net pass: {arr["avg_nap_net_pass"]:.4f}')
    plt.axhline(y=arr["avg_nap_net_pass"], linestyle='-')
    plt.xlabel("N known patterns to compare with")
    plt.ylabel("Execution time (seconds)")
    plt.legend(loc='upper left')
    plt.title("Server CIFAR10")
    plt.savefig("results/article_plots/execution_times/ServerCIFAR10")

    # show_values_on_bars(axes)


def compare_exec_times_all_methods():
    # d1_tasks = ['MNIST']
    d1_tasks = ['MNIST', "FashionMNIST", "CIFAR10", "CIFAR100", "STL10", "TinyImagenet"]
    data = []
    for d1 in d1_tasks:

        file_pattern = "*" + d1 + '*.npz'
        files = glob.glob(os.path.join("results/article_plots/execution_times", file_pattern))
        for file in files:
            if d1 == "CIFAR10":
                print(file)
                if file.split('/')[-1].split("_")[2] == "CIFAR100":

                    continue
            method = file.split("/")[-1].split("_")[0]
            model = file.split("/")[-1].split("_")[1]
            exec_time = np.load(file)["exec_times"]
            data.append((method, model, d1, exec_time.item()))
    df = pd.DataFrame(data, columns=["method", "model", "dataset", "exec_time"])
    print(df)
    # _ = sns.catplot(x="method", y="exec_time", kind="box", data=df)
    # plt.show()
    grouped = df.groupby(["method", "model"])[
        "exec_time"].mean().sort_values(ascending=True)
    figure, axes = plt.subplots()
    x = np.arange(len(grouped.index))
    v = axes.bar(x, grouped.values)
    show_values_on_bars(axes)
    plt.xticks(x, grouped.index, rotation=90)
    plt.xlabel("Method, architecture")
    plt.ylabel("time (s)")
    plt.tight_layout()
    plt.title("Mean execution time per one sample")
    plt.show()
    # plt.savefig("results/article_plots/execution_times/plots/" + d1)


def auroc():
    d1_tasks = ['MNIST', 'FashionMNIST', 'STL10', "CIFAR100"]

    d2_tasks = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'STL10', 'CIFAR100',
                'TinyImagenet']
    d3_tasks = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'STL10', 'CIFAR100',
                'TinyImagenet']
    layer_dict = {
        "MNIST": 6,
        "FashionMNIST": 2,
        "CIFAR10": 4,
        "CIFAR100": 4,
        "STL10": 11,
        "TinyImagenet": 6
    }
    scores = 0
    auroc_sum = 0
    aupr_sum = 0
    acc_sum = 0
    counter = 0
    model_name = "VGG"
    for d1 in d1_tasks:
        best_layer = layer_dict[d1]

        for d2 in d2_tasks:
            if d2 in d2_compatiblity[d1]:
                best_auroc = 0
                best_acc = 0
                best_aupr = 0
                best_auroc_pool_type = 0
                best_acc_pool_type = 0
                best_aupr_pool_type = 0
                best_acc_layer = 0
                best_aupr_layer = 0
                best_auroc_layer = 0
                for pool_type in ["avg", "max"]:
                    for layer in range(len(layers_shapes[model_name][d1])):
                        frames = []
                        for d3 in d3_tasks:
                            if d2 != d3 and d3 in d2_compatiblity[d1]:

                                file_pattern = model_name + '_' + d1 + '_' + d2 + '_' + d3 + "_" + str(
                                    layer) + "_" + pool_type + ".csv"
                                files = glob.glob(
                                    os.path.join("results/article_plots/full_nets/fixed", file_pattern))

                                for file in files:
                                    # print(file)
                                    df = pd.read_csv(file, index_col=0)
                                    rows = len(df.index)
                                    df["label"] = 0
                                    df.loc[int(rows / 2):, "label"] = 1
                                    frames.append(df)
                                    # print(f'{df["correct"].sum() / len(df.index)}')
                                    # print(df)
                        # print(f"{model_name} {d1} vs {d2} layer {layer}")
                        frame = pd.concat(frames, axis=0, ignore_index=True)

                        score = roc_auc_score(frame["label"], frame["distance"])
                        acc = frame["correct"].sum() / len(frame.index)
                        lr_precision, lr_recall, _ = precision_recall_curve(frame["label"], frame["distance"])
                        lr_auc = auc(lr_recall, lr_precision)
                        if score > best_auroc:
                            best_auroc_layer = layer
                            best_auroc = score
                            best_auroc_pool_type = pool_type
                        if acc > best_acc:
                            best_acc_layer = layer
                            best_acc = acc
                            best_acc_pool_type = pool_type
                        if lr_auc > best_aupr:
                            best_aupr_layer = layer
                            best_aupr = lr_auc
                            best_aupr_pool_type = pool_type
                        # print(f"DT: {d1} auroc: {score}")
                        # print(f"DT: {d1} acc: {acc}")
                auroc_sum += best_auroc
                aupr_sum += best_aupr
                acc_sum += best_acc
                counter += 1
                print(
                    f"{model_name} {d1} vs {d2} best auroc layer {best_auroc_layer} pt {best_auroc_pool_type} auroc {best_auroc}"
                    f" best aupr layer {best_aupr_layer} aupr {best_aupr} pt {best_aupr_pool_type} "
                    f" best acc layer {best_acc_layer} acc {best_acc} pt {best_acc_pool_type}")

        # counter += 1
        # scores += score
    print(f"Aggregated auroc: {auroc_sum / counter} aupr: {aupr_sum / counter} acc: {acc_sum / counter}")


def choose_layers_and_pool_type(thresholds, accuracies, model, dt, type=0, votes=1, steps=5, thresholds_factor=0.1):
    linspace = np.linspace(0.1, 0.9, steps)
    quantile_factors = np.sqrt(1. / np.abs(linspace - np.rint(linspace)))
    max_threshold = np.max((thresholds + quantile_factors) * quantile_factors, axis=2)[:, :, np.newaxis]
    scores = (accuracies - 0.5) * (
            thresholds_factor + np.abs(
        ((thresholds + quantile_factors) * quantile_factors - max_threshold) / max_threshold))
    max_acc_ids = np.argmax(scores, axis=2)[:, :, np.newaxis]
    best_thresholds = np.take_along_axis(thresholds, max_acc_ids, axis=2).squeeze()
    best_accuracies = np.take_along_axis(accuracies, max_acc_ids, axis=2).squeeze()
    new_th = np.zeros(thresholds.shape[1])
    new_acc = np.zeros(thresholds.shape[1])
    chosen = []

    shapes = np.array(layers_shapes[model][dt])
    shape_factors = shapes / shapes.min()
    max_factor = shape_factors.max()

    for layer_id in range(thresholds.shape[1]):
        max_threshold_pools = np.min(max_threshold[:, layer_id, :])
        scores = (best_accuracies[:, layer_id] - 0.5) * (thresholds_factor + np.abs(
            ((best_thresholds[:, layer_id] + quantile_factors[max_acc_ids[:, layer_id, :]]) * quantile_factors[
                max_acc_ids[:, layer_id, :]] - max_threshold_pools) / max_threshold_pools))

        if type == 0:
            if (scores[:, 0] >= scores[:, 1]).all():
                add_factor = quantile_factors[max_acc_ids[0, layer_id, :]] + shape_factors[layer_id]
                multiplier = quantile_factors[max_acc_ids[0, layer_id, :]] * (max_factor / shape_factors[layer_id])
                chosen.append((layer_id, "max", add_factor, multiplier))
                new_th[layer_id] = best_thresholds[0, layer_id]
                new_acc[layer_id] = best_accuracies[0, layer_id]
            else:
                add_factor = quantile_factors[max_acc_ids[1, layer_id, :]] + shape_factors[layer_id]
                multiplier = quantile_factors[max_acc_ids[1, layer_id, :]] * (max_factor / shape_factors[layer_id])
                chosen.append((layer_id, "avg", add_factor, multiplier))
                new_th[layer_id] = best_thresholds[1, layer_id]
                new_acc[layer_id] = best_accuracies[1, layer_id]
        elif type == 1:
            if scores[0, 0] >= scores[1, 1]:
                add_factor = quantile_factors[max_acc_ids[0, layer_id, :]] + shape_factors[layer_id]
                multiplier = quantile_factors[max_acc_ids[0, layer_id, :]] * (max_factor / shape_factors[layer_id])

                chosen.append((layer_id, "max", add_factor, multiplier))
                new_th[layer_id] = best_thresholds[0, layer_id]
                new_acc[layer_id] = best_accuracies[0, layer_id]
            else:
                add_factor = quantile_factors[max_acc_ids[1, layer_id, :]] + shape_factors[layer_id]
                multiplier = quantile_factors[max_acc_ids[1, layer_id, :]] * (max_factor / shape_factors[layer_id])

                chosen.append((layer_id, "avg", add_factor, multiplier))
                new_th[layer_id] = best_thresholds[1, layer_id]
                new_acc[layer_id] = best_accuracies[1, layer_id]
        elif type == 2:
            if best_accuracies[0, layer_id] >= best_accuracies[1, layer_id]:
                add_factor = quantile_factors[max_acc_ids[0, layer_id, :]] + shape_factors[layer_id]
                multiplier = quantile_factors[max_acc_ids[0, layer_id, :]] * (max_factor / shape_factors[layer_id])

                chosen.append((layer_id, "max", add_factor, multiplier))
                new_th[layer_id] = best_thresholds[0, layer_id]
                new_acc[layer_id] = best_accuracies[0, layer_id]
            else:
                add_factor = quantile_factors[max_acc_ids[1, layer_id, :]] + shape_factors[layer_id]
                multiplier = quantile_factors[max_acc_ids[1, layer_id, :]] * (max_factor / shape_factors[layer_id])

                chosen.append((layer_id, "avg", add_factor, multiplier))
                new_th[layer_id] = best_thresholds[1, layer_id]
                new_acc[layer_id] = best_accuracies[1, layer_id]
        elif type == 3:
            if best_thresholds[0, layer_id] < best_thresholds[1, layer_id]:
                add_factor = quantile_factors[max_acc_ids[0, layer_id, :]] + shape_factors[layer_id]
                multiplier = quantile_factors[max_acc_ids[0, layer_id, :]] * (max_factor / shape_factors[layer_id])

                chosen.append((layer_id, "max", add_factor, multiplier))
                new_th[layer_id] = best_thresholds[0, layer_id]
                new_acc[layer_id] = best_accuracies[0, layer_id]
            else:
                add_factor = quantile_factors[max_acc_ids[1, layer_id, :]] + shape_factors[layer_id]
                multiplier = quantile_factors[max_acc_ids[1, layer_id, :]] * (max_factor / shape_factors[layer_id])

                chosen.append((layer_id, "avg", add_factor, multiplier))
                new_th[layer_id] = best_thresholds[1, layer_id]
                new_acc[layer_id] = best_accuracies[1, layer_id]

    chosen_layers = new_acc.argsort()[::-1][:votes]

    chosen = np.array(chosen, dtype=object)
    return chosen[chosen_layers]


def fixed():
    d1_tasks = ['MNIST', 'FashionMNIST', 'STL10', 'CIFAR100', "CIFAR10", "TinyImagenet"]
    d2_tasks = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'STL10', 'CIFAR100',
                'TinyImagenet']
    d3_tasks = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'STL10', 'CIFAR100',
                'TinyImagenet']
    types = [0, 1, 2, 3]
    n_votes = [3, 5, 7, 9]
    th_factors = [0.1]
    data = []
    for model_name in ["VGG", "Resnet"]:
        for type in types:
            for votes in n_votes:
                for tf in th_factors:
                    agg_acc = 0
                    agg_acc2 = 0
                    agg_acc3 = 0
                    agg_auroc = 0
                    agg_aupr = 0
                    counter = 0
                    for d1 in d1_tasks:
                        for d2 in d2_tasks:
                            if d2 in d2_compatiblity[d1]:
                                df_thresholds = dict()
                                df_thresholds["max"] = pd.read_csv(
                                    "results/article_plots/full_nets/fixed/hamming/" + model_name + '_' + d1 + '_' + d2 + 'maxth-acc.csv',
                                    index_col=0)
                                df_thresholds["avg"] = pd.read_csv(
                                    "results/article_plots/full_nets/fixed/hamming/" + model_name + '_' + d1 + '_' + d2 + 'avgth-acc.csv',
                                    index_col=0)
                                np_file = np.load(
                                    "results/article_plots/full_nets/fixed/hamming/" + model_name + '_' + d1 + '_' + d2 + '.npz')
                                np_accuracies = np_file["accuracies"]
                                np_thresholds = np_file["thresholds"]
                                for d3 in d3_tasks:
                                    if d2 != d3 and d3 in d2_compatiblity[d1]:
                                        file_pattern = model_name + '_' + d1 + '_' + d2 + '_' + d3 + "_*"
                                        files = glob.glob(
                                            os.path.join("results/article_plots/full_nets/fixed/hamming", file_pattern))
                                        frames = dict()
                                        frames["max"] = dict()
                                        frames["avg"] = dict()
                                        rows = 0

                                        for file in files:
                                            df = pd.read_csv(file, index_col=0)
                                            rows = len(df.index)
                                            df["label"] = 0
                                            df.loc[int(rows / 2):, "label"] = 1
                                            layernum = file.split("_")[-2]
                                            pool_type = file.split("_")[-1].split(".")[0]
                                            if df_thresholds[pool_type]["threshold"][int(layernum)] != -1:
                                                frames[pool_type][layernum] = df
                                        correct_count = 0

                                        chosen_ids = choose_layers_and_pool_type(thresholds=np_thresholds,
                                                                                 accuracies=np_accuracies, type=type,
                                                                                 votes=votes, thresholds_factor=tf,
                                                                                 model=model_name, dt=d1)
                                        # for (chosen_layer, chosen_pool_type) in chosen_ids:
                                        #     print(chosen_layer, chosen_pool_type)
                                        distances_1 = np.zeros((rows, len(chosen_ids)))
                                        distances_2 = np.zeros((rows, len(chosen_ids)))
                                        distances_3 = np.zeros((rows, len(chosen_ids)))
                                        labels = np.zeros(rows)
                                        labels[int(rows / 2):] = 1
                                        for i in range(rows):
                                            correct_votes = 0
                                            for vote_id, (
                                            chosen_layer, chosen_pool_type, add_factor, multiplier) in enumerate(
                                                    chosen_ids):
                                                scaled_th = (df_thresholds[chosen_pool_type]["threshold"][
                                                                 chosen_layer] + add_factor) * multiplier
                                                distances_1[i, vote_id] = (frames[chosen_pool_type][str(chosen_layer)][
                                                                               "distance"][i] + add_factor) * multiplier
                                                distances_2[i, vote_id] = (frames[chosen_pool_type][str(chosen_layer)][
                                                                               "distance"][
                                                                               i] + add_factor) * multiplier - scaled_th
                                                distances_3[i, vote_id] = (frames[chosen_pool_type][str(chosen_layer)][
                                                                               "distance"][
                                                                               i] + add_factor) * multiplier / scaled_th
                                                correct_votes += frames[chosen_pool_type][str(chosen_layer)]["correct"][
                                                    i]
                                            correct_count += (correct_votes > (len(chosen_ids) / 2))
                                        distances_1 = distances_1.sum(axis=1)
                                        distances_2 = distances_2.sum(axis=1)
                                        distances_3 = distances_3.sum(axis=1)

                                        acc = correct_count / rows
                                        acc2 = (distances_2[:int(rows / 2)] <= 0).sum() / rows
                                        acc2 += (distances_2[int(rows / 2):] > 0).sum() / rows
                                        acc3 = (distances_3[:int(rows / 2)] <= votes).sum() / rows
                                        acc3 += (distances_3[int(rows / 2):] > votes).sum() / rows

                                        auroc_score_1 = roc_auc_score(labels, distances_1)
                                        auroc_score_2 = roc_auc_score(labels, distances_2)
                                        auroc_score_3 = roc_auc_score(labels, distances_3)
                                        # print(distances_3)
                                        # plt.boxplot(distances_3[:int(rows / 2)])
                                        # plt.show()
                                        # plt.boxplot(distances_3[int(rows / 2):])
                                        # plt.show()
                                        # exit(0)
                                        lr_precision_1, lr_recall_1, _ = precision_recall_curve(labels, distances_1)
                                        lr_precision_2, lr_recall_2, _ = precision_recall_curve(labels, distances_2)
                                        lr_precision_3, lr_recall_3, _ = precision_recall_curve(labels, distances_3)
                                        aupr_1 = auc(lr_recall_1, lr_precision_1)
                                        aupr_2 = auc(lr_recall_2, lr_precision_2)
                                        aupr_3 = auc(lr_recall_3, lr_precision_3)
                                        agg_acc += acc
                                        agg_acc2 += acc2
                                        agg_acc3 += acc3
                                        agg_auroc += auroc_score_1
                                        agg_aupr += aupr_1
                                        data.append((
                                                    model_name, d1, d2, d3, type, votes, tf, auroc_score_1, aupr_1, acc,
                                                    acc2, acc3, auroc_score_2, aupr_2, auroc_score_3, aupr_3))
                                        counter += 1

                        # print(f"{model_name}  {d1} - type {type}, votes {votes}, thfactor {tf} Aggregated accuracy: {agg_acc / counter}")
                        print(
                            f"{model_name}  {d1} - type {type}, votes {votes}, thfactor {tf} Aggregated auroc: {agg_auroc / counter}"
                            f"Aggregated aupr: {agg_aupr / counter} acc: {agg_acc / counter} acc2: {agg_acc2 / counter} acc3: {agg_acc3 / counter}")
    pd.DataFrame(data,
                 columns=["model", "d1", "d2", "d3", "type", "votes", "tf", "auroc1", "aupr1", "acc", "acc2", "acc3",
                          "auroc2", "aupr2", "auroc3", "aupr3"]).to_csv("hamming_redoall.csv")


def choose_best_auroc(model="Resnet"):
    results = pd.read_csv("hamming_redoall.csv", index_col=0)
    # results1 = pd.read_csv("hamming_redo.csv", index_col=0)
    # results2 = pd.read_csv("hamming_aurocexp.csv", index_col=0)
    # results = pd.concat([results1, results2], axis=0, ignore_index=True)
    grouped = results.groupby(["type", "votes", "tf"])[
        "auroc1", "aupr1", "acc", "acc2", "acc3" , "auroc2", "aupr2", "auroc3", "aupr3"].mean()
    # print(grouped.sort_values("acc").tail(10))

    # print(grouped.sort_values("auroc3").tail(3))

    # print(grouped.sort_values("aupr3").tail(3))
    results = results[results["model"] == model]
    # results = results[results["d1"] == "FashionMNIST"]
    grouped = results.groupby(["type", "votes", "tf"])[["acc", "acc2", "acc3" , "auroc3", "auroc1", "auroc2", ]].mean()
    # grouped = grouped[grouped["acc"] > 0.823]
    # grouped = grouped[grouped["auroc3"] > 0.884]
    # grouped = grouped[grouped["aupr3"] > 0.879]
    print(grouped)
    # print(grouped.sort_values("auroc1").tail(3))
    # print(grouped.sort_values("auroc2").tail(3))
    # print(grouped.sort_values("auroc3").tail(3))
    # for type in [0, 1, 2, 3]:
    #     results_type = results[results["type"] == type]
    #     for model in ["VGG", "Resnet"]:
    #         results_model = results_type[results_type["model"] == model]
    #         for datasets in [["MNIST", "FashionMNIST"], ["CIFAR10", "CIFAR100", "STL10", "TinyImagenet"]]:
    #             results_datasets = results_model[results_model["d1"].isin(datasets)]
    #             grouped = results_datasets.groupby(["votes", "tf"])[["acc", "auroc3", "aupr3"]].mean()
    #             print(f"type: {type}  model {model}  datasets {datasets}")
    #             print(grouped.sort_values("acc").tail(3))
    #             print(grouped.sort_values("auroc3").tail(3))
    #             print(grouped.sort_values("aupr3").tail(3))


def nap_model_to_method(x, start=0):
    if x == "VGG":
        return "nap/" + str(start)
    return "nap/" + str(start + 1)



def add_method_name_column(fname=None, df=None):
    names = {
        "nap/0": "ActivationPatterns_acc_9_1/VGG",
        "nap/1": "ActivationPatterns_acc_9_1/Res",
        "nap/2": "ActivationPatterns_acc_9_3/VGG",
        "nap/3": "ActivationPatterns_acc_9_3/Res",
        "nap/4": "ActivationPatterns_thresh_5_1/VGG",
        "nap/5": "ActivationPatterns_thresh_5_1/Res",
        "nap/6": "ActivationPatterns_thresh_5_3/VGG",
        "nap/7": "ActivationPatterns_thresh_5_3/Res",
        "outlier_exposure/0": "OutlierExposure/VGG",
        "outlier_exposure/1": "OutlierExposure/Res",
        "grad_norm/0": "GradNorm/VGG",
        "grad_norm/1": "GradNorm/Res",
        "react/0": "ReAct/VGG",
        "react/1": "ReAct/Res",
        "energy/0": "Energy/VGG",
        "energy/1": "Energy/Res",
        "mahalanobis/0": "Mahalanobis/VGG",
        "mahalanobis/1": "Mahalanobis/Res",
        "msad/0": "MeanShiftedAD/VGG",
        "msad/1": "MeanShiftedAD/Res",
        "odin/0": "ODIN/VGG",
        "odin/1": "ODIN/Res",
        "score_svm/0": "ScoreSVM/VGG",
        "score_svm/1": "ScoreSVM/Res",
        "logistic_svm/0": "Log.SVM/VGG",
        "logistic_svm/1": "Log.SVM/Res",
        "reconst_thresh/0": "AEThre./BCE",
        "reconst_thresh/1": "AEThre./MSE",
        "mcdropout/0": "MC-Dropout",
        "knn/1": "1-NNSVM",
        "knn/2": "2-NNSVM",
        "knn/4": "4-NNSVM",
        "knn/8": "8-NNSVM",
        "bceaeknn/1": "1-BNNSVM",
        "bceaeknn/2": "2-BNNSVM",
        "bceaeknn/4": "4-BNNSVM",
        "bceaeknn/8": "8-BNNSVM",
        "mseaeknn/1": "1-MNNSVM",
        "mseaeknn/2": "2-MNNSVM",
        "mseaeknn/4": "4-MNNSVM",
        "mseaeknn/8": "8-MNNSVM",
        "vaeaeknn/1": "1-VNNSVM",
        "vaeaeknn/2": "2-VNNSVM",
        "vaeaeknn/4": "4-VNNSVM",
        "vaeaeknn/8": "8-VNNSVM",
        "deep_ensemble/0": "DeepEns./VGG",
        "deep_ensemble/1": "DeepEns./Res",
        "prob_threshold/0": "PbThresh/VGG",
        "prob_threshold/1": "PbThresh/Res",
        "binclass/0": "BinClass/VGG",
        "binclass/1": "BinClass/Res",
        "openmax/0": "OpenMax/VGG",
        "openmax/1": "OpenMax/Res",
        "pixelcnn/0": "PixelCNN++"
    }
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(fname, index_col=0)
    df['method'] = df.apply(lambda row: names[row.m], axis=1)
    df.to_csv(fname)


def show_values_on_bars(axs, ):
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.3f}'.format(p.get_height())
            ax.text(_x, _y + 0.01, value, ha="center", rotation=90)

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

def configuration_experiment():
    d1_tasks = ['MNIST', 'FashionMNIST', 'STL10', 'CIFAR100', "CIFAR10", "TinyImagenet"]
    d2_tasks = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'STL10', 'CIFAR100',
                'TinyImagenet']
    d3_tasks = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'STL10', 'CIFAR100',
                'TinyImagenet']
    for model_name in ["VGG", "Resnet"]:
        counter = 0
        acc_sum = 0
        auroc_sum = 0
        acc_sum2 = 0
        auroc_sum2 = 0
        b_acc_sum = 0
        b_auroc_sum = 0
        b_c = 0

        for d1 in d1_tasks:
            d1_counter = 0
            d1_acc_sum = 0
            d1_auroc_sum = 0
            frames = []
            for d2 in d2_tasks:
                if d2 in d2_compatiblity[d1]:

                    for d3 in d3_tasks:
                        if d2 != d3 and d3 in d2_compatiblity[d1]:
                            file_pattern = model_name + d1 + d2 + d3 + "*otherlayers*"
                            files = glob.glob(
                                os.path.join("results/article_plots", file_pattern))
                            for file in files:
                                df = pd.read_csv(file, index_col=0)
                                b_acc_sum += df["test_acc"].max()
                                b_auroc_sum += df["auroc"].max()
                                b_c += 1
                                frames.append(df)
            frame = pd.concat(frames, axis=0, ignore_index=True)
            grouped = frame.groupby(["model", "ds", "layer", "pool_type", "quantile"])[
                ["test_acc", "auroc", "aupr"]].mean()
            grouped2 = frame.groupby(["model", "ds", "dv", "layer", "pool_type", "quantile"])[
                ["test_acc", "auroc", "aupr"]].mean()
            acc_sum += grouped.sort_values("test_acc").tail(1)["test_acc"].values.item()
            auroc_sum += grouped.sort_values("auroc").tail(1)["auroc"].values.item()
            acc_sum2 += grouped2.sort_values("test_acc").tail(1)["test_acc"].values.item()
            auroc_sum2 += grouped2.sort_values("auroc").tail(1)["auroc"].values.item()
            counter += 1
            d1_acc_sum += grouped.sort_values("test_acc").tail(1)["test_acc"].values.item()
            d1_auroc_sum += grouped.sort_values("auroc").tail(1)["auroc"].values.item()
            d1_counter += 1
            # print(f" {model_name} {d1} max_test_acc: {d1_acc_sum/d1_counter} auroc {d1_auroc_sum/d1_counter} l {grouped.sort_values('test_acc').tail(1)}")

        print(f" {model_name} bmax_test_acc: {b_acc_sum / b_c} b2max_test_acc: {acc_sum2 / counter} b3max_test_acc: {acc_sum / counter}"
              f"  bauroc {b_auroc_sum / b_c} b2auroc {auroc_sum2 / counter} b3auroc {auroc_sum / counter} ")
if __name__ == "__main__":

    # from methods.nap.cuda_tree.nonrecursive_cython import BallTree
    # data = np.array([[1,1,1,1,1,1], [1,1,1,0,0,0], [0,0,0,1,1,1], [0,0,0,0,0,0]])
    # b = BallTree(data)
    # print(b.query([[0,1,1,0,1,0]]))
    # exit(0)
    # draw_boxplots()
    # results = pd.read_csv("results/results_working_methods.csv", index_col=0)
    # results2 = pd.read_csv("results/results_fixed_methods.csv", index_col=0)
    # results3 = pd.read_csv("results/results_resnet_all_datasets.csv", index_col=0)
    # results4 = pd.read_csv("results/results_vgg_all_datasets.csv", index_col=0)
    # save_results_as_csv(pd.concat([results, results2, results3, results4]), "results/results_all.csv")
    # draw("results/results_all.csv")
    # df["acc"] = df["acc"].apply(lambda x: eval(x.split("[")[1].split("]")[0]))
    # df.to_csv("nap_confirm2.csv")
    # fixed()
    # exit(0)
    # df = pd.read_csv("allmethods_auroc2.csv", index_col=0)
    # df2 = pd.read_csv("hamming_redoall.csv", index_col=0)
    # df3 = df2.copy()
    # df2 = df2[df2["tf"] == 0.1]
    # df2 = df2[df2["votes"] == 9]
    # df2 = df2[df2["type"] == 2]
    # df2_3 = df2.copy()
    # df2.drop(["type", "votes", "tf", "auroc1", "aupr1", "auroc3", "aupr3", "aupr2", "acc3", "acc"], inplace=True, axis=1)
    # df2.model = df2.model.apply(lambda x: nap_model_to_method(x))
    # df2.rename(
    #     columns={"model": "m", "auroc2": "auroc", "acc2": "acc", "d1": "ds", "d2": "dv", "d3": "dt"},
    #     inplace=True)
    # add_method_name_column(df=df2)
    # df2_3.model = df2_3.model.apply(lambda x: nap_model_to_method(x, 2))
    # df2_3.drop(["type", "votes", "tf", "auroc1", "aupr1", "auroc3", "aupr3", "aupr2", "acc3", "acc2"], inplace=True, axis=1)
    # df2_3.rename(
    #     columns={"model": "m", "auroc2": "auroc", "d1": "ds", "d2": "dv", "d3": "dt"},
    #     inplace=True)
    # add_method_name_column(df=df2_3)
    #
    # df3 = df3[df3["tf"] == 0.1]
    # df3 = df3[df3["votes"] == 5]
    # df3 = df3[df3["type"] == 3]
    # df3_3 = df3.copy()
    # df3.drop(["type", "votes", "tf", "auroc1", "aupr1", "auroc3", "aupr3", "aupr2", "acc3", "acc"], inplace=True, axis=1)
    # df3.model = df3.model.apply(lambda x: nap_model_to_method(x, 4))
    # df3.rename(
    #     columns={"model": "m", "auroc2": "auroc", "acc2": "acc", "d1": "ds", "d2": "dv", "d3": "dt"},
    #     inplace=True)
    # add_method_name_column(df=df3)
    # df3_3.model = df3_3.model.apply(lambda x: nap_model_to_method(x, 6))
    # df3_3.drop(["type", "votes", "tf", "auroc1", "aupr1", "auroc3", "aupr3", "aupr2", "acc3", "acc2"], inplace=True, axis=1)
    # df3_3.rename(
    #     columns={"model": "m", "auroc2": "auroc", "d1": "ds", "d2": "dv", "d3": "dt"},
    #     inplace=True)
    # add_method_name_column(df=df3_3)


    # df2.model = df2.model.apply(lambda x: nap_model_to_method(x))
    # df2.drop(["type", "votes", "tf", "auroc1", "aupr1", "auroc3", "aupr3", "acc", "acc3"], inplace=True, axis=1)
    # print(df2)
    # df2.rename(columns={"model": "m", "auroc2": "auroc", "aupr2": "aupr", "acc2": "acc", "d1": "ds", "d2": "dv", "d3": "dt"}, inplace=True)
    # print(df2)
    # d = pd.concat([df, df2, df3, df2_3], axis=0, ignore_index=True)
    # d = d.loc[:, ~d.columns.str.contains('^Unnamed')]
    # d = d.drop(["method", "a"], axis=1)
    # d.loc[d["m"] == "grad_norm/1", "auroc"] = 1 - d[d["m"] == "grad_norm/1"]["auroc"]
    # d.loc[d["m"] == "grad_norm/0", "auroc"] = 1 - d[d["m"] == "grad_norm/0"]["auroc"]
    # d = d[d["m"] != "nap/3"]
    # d = d[d["m"] != "nap/4"]
    # d.to_csv("allmethods_auroc_twonaps.csv")

    # add_method_name_column("allmethods_auroc.csv")



    # draw("allmethods_auroc_twonaps.csv", "acc", "Mean test accuracy")




    # draw_article_plots("FashionMNIST", model="VGG", q=0.9)
    # generate_latex('matplotlib_ex-redo', r'1\textwidth', dpi=100)
    # generate_latex_heatmaps('matplotlib_ex-heatmaps-redo', r'1\textwidth', dpi=100)
    # fix_vgg_results()
    # full_net_plot()

    # choose_best_auroc()
    # choose_best_auroc("VGG")
    # execution_times_plot()
    # compare_exec_times_all_methods()
    # results = torch.load("workspace/experiments/simple-eval/results.pth")
    # df = pd.DataFrame(results, columns=["m", "ds", "dv", "dt", "method", "a", "acc", "auroc", "aupr"])
    # df = df.drop(["method", "a"], axis=1)
    # df = pd.read_csv("single_neuron_exp.csv", index_col=0)
    # d2 = df[df["m"] == "nap/1"]
    # print(d2["acc"].mean())
    # print(d2["auroc"].mean())
    # d2 = df[df["m"] == "nap/0"]
    # print(d2["acc"].mean())
    # print(d2["auroc"].mean())
    # df.to_csv("single_neuron_exp.csv")
    # print(results)
    plt.rcParams.update({'font.size': 12})
    # # draw_hamming_distances_layerwise()
    #
    # draw_article_plots("FashionMNIST", "MNIST", "VGG")
    # draw_article_plots("FashionMNIST", "NormalNoise", "VGG")
    # draw_article_plots("FashionMNIST", "NormalNoise", "VGG", q=0.9)
    # draw_article_plots("FashionMNIST", "MNIST", "VGG", q=0.9)

    # draw_article_plots("FashionMNIST", model="VGG")
    # draw("allmethods_auroc_twonaps2.csv", "acc", label="Uśredniona celność testowa")
    # draw("allmethods_auroc_onenap.csv", "auroc", label="Uśredniona metryka AUROC")

    # compare_exec_times_all_methods()
    configuration_experiment()