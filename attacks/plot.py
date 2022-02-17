import os.path
import glob
import pandas as pd
import matplotlib.pyplot as plt


def draw_hamming_distances():
    for layer_num in range(4):
        train_file_pattern = "attacks" + str(layer_num) + "_0.0"
        test_file_pattern = "sameclass" + str(layer_num) + "_0.0"
        train_files = glob.glob(os.path.join("results", train_file_pattern))
        test_files = glob.glob(os.path.join("results", test_file_pattern))

        li = []
        for filename in train_files:
            df = pd.read_csv(filename, index_col=0)
            li.append(df)

        frame = pd.concat(li, axis=0, ignore_index=True)
        for filename in test_files:
            df = pd.read_csv(filename, index_col=0)
            frame_train_sampled = frame
            df = df.sample(len(frame.index))
            title = "AttacksVSSameClass_layer" + str(layer_num)
            plt.figure()
            _ = plt.hist(frame_train_sampled, bins=int(10), alpha=0.7, label='attack')
            _ = plt.hist(df, bins=int(10), alpha=0.7, label='same_class')
            plt.legend(loc='lower right')
            plt.title(title)
            plt.show()
            # plt.savefig(os.path.join("results/plots", title))
            plt.close()


if __name__ == "__main__":
    draw_hamming_distances()
