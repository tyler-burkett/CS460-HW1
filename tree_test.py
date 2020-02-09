from decision_tree import DecisionTree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

synthetic_labels = 2
plot_colors = "br"
plot_inc = 0.1

if __name__ == "__main__":
    # Move to data directory
    cwd = os.getcwd()
    os.chdir(cwd + "\\data")

    for i in range(1, 5):
        # Read in synth_data
        synth_data = pd.read_csv("synthetic-{}.csv".format(i),
                                names=["f1", "f2", "label"],
                                dtype={"f1": "float64",
                                    "f2": "float64",
                                    "label": "category"})
        print("synthetic-{} data".format(i))

        # Determine minimum bin_size that yields best accuracy
        best_score = (0, 0, None)
        for num_bins in range(2, 50):
            tree = DecisionTree(num_bins)
            tree.fit(synth_data, limit=3)
            results = tree.predict(synth_data)
            num_correct = sum(row["label"] == synth_data.at[index, "label"]
                                for index, row in results.iterrows())
            accuracy = num_correct / len(synth_data)
            print("accuracy (bins = {}): {}".format(num_bins, accuracy))
            if best_score[1] < accuracy:
                best_score = (num_bins, accuracy, tree)
                if accuracy == 1.0:
                    break
        print("best tree: bins = {}, accuracy = {}".format(*best_score))

        # Plot decision surface of best decision tree for current data;
        # Following example plots from scikitlearn:
        # https://scikit-learn.org/0.15/auto_examples/tree/plot_iris.html
        plt.subplot(2, 2, i)

        x_min, x_max = synth_data["f1"].min() - 1, synth_data["f1"].max() + 1
        y_min, y_max = synth_data["f2"].min() - 1, synth_data["f2"].max() + 1
        x_list, y_list = np.meshgrid(np.arange(x_min, x_max, plot_inc),
                                     np.arange(y_min, y_max, plot_inc))
        print(np.c_[x_list.ravel(), y_list.ravel()])
        plot_grid = best_score[2].predict(pd.DataFrame(np.c_[x_list.ravel(), y_list.ravel()],
                                                        columns=["f1", "f2"]))
        plot_grid = plot_grid.to_numpy().reshape(x_list.shape)
        print(plot_grid)
        plt.contourf(x_list, y_list, plot_grid, cmap=plt.cm.Paired)
        plt.xlabel = "f1"
        plt.ylabel = "f2"
        plt.axis("tight")
        for i, color in zip(range(synthetic_labels), plot_colors):
            print(synth_data["label"])
            idx = synth_data[synth_data["label"] == synth_data["label"].dtype.categories[i]]
            print(idx)
            plt.scatter(idx["f1"], idx["f2"],
                        c=color, label=i,
                        cmap=plt.cm.Paired)
        plt.legend()
        plt.show()
