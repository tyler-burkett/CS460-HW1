import pandas as pd
import pandas_util as pdu
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
from decision_tree import DecisionTree
from anytree import RenderTree

# Values used when making plots
synthetic_labels = 2
plot_colors = "br"
plot_inc = 0.1
binary_cmap = colors.LinearSegmentedColormap.from_list("", ["#9999ff", "#ff9999"], N=2)

# Explicit data types of features in data sets
synth_data_names = ["f1", "f2", "label"]
synth_data_types = {"f1": "float64",
                    "f2": "float64",
                    "label": "category"}
vd_data_types = {"Platform": "category",
                 "Genre": "category",
                 "Publisher": "category",
                 "Developer": "category",
                 "Rating": "category"}

if __name__ == "__main__":
    # Move to data directory
    cwd = os.getcwd()
    os.chdir(cwd + "\\data")

    for i in range(1, 5):
        # Read in synth_data
        synth_data = pd.read_csv("synthetic-{}.csv".format(i),
                                names=synth_data_names,
                                dtype=synth_data_types)
        print("synthetic-{} data".format(i))

        # Train tree (default bin values of 10)
        tree = DecisionTree(10)
        tree.fit(synth_data, limit=3)
        print(RenderTree(tree.root))
        results = tree.predict(synth_data)
        num_correct = sum(row["label"] == synth_data.at[index, "label"]
                            for index, row in results.iterrows())
        accuracy = num_correct / len(synth_data)
        print("accuracy: {}".format(accuracy))

        # Plot decision surface of best decision tree for current data;
        # Following example plots from scikitlearn:
        # https://scikit-learn.org/0.15/auto_examples/tree/plot_iris.html

        # Make subplot
        plt.subplot(2, 2, i, title="synthetic-{} data".format(i))

        # Form meshgrid for surface map of decision tree
        x_min, x_max = synth_data["f1"].min(), synth_data["f1"].max()
        y_min, y_max = synth_data["f2"].min(), synth_data["f2"].max()
        x_list, y_list = np.meshgrid(np.arange(x_min, x_max, plot_inc),
                                     np.arange(y_min, y_max, plot_inc))

        # Predict labels for values of meshgrid
        plot_grid = tree.predict(pd.DataFrame(np.c_[x_list.ravel(), y_list.ravel()],
                                                        columns=["f1", "f2"]))
        plot_grid = plot_grid.iloc[:, -1].to_numpy().reshape(x_list.shape)

        # Plot surfacemap of decision tree predictions
        plt.contourf(x_list, y_list, plot_grid, cmap=binary_cmap)
        plt.xlabel("f1")
        plt.ylabel("f2")
        plt.axis("tight")

        # Plot testing points over surface map to compare actual points labels
        # to tree predicitons
        for i, color in zip(range(synthetic_labels), plot_colors):
            idx = synth_data[synth_data["label"] == synth_data["label"].dtype.categories[i]]
            plt.scatter(idx["f1"], idx["f2"], s=5,
                        c=color, label=i,
                        cmap=plt.cm.Paired)
        plt.legend()
    # Show plots for synthetic data
    plt.show()

    # Video Game Data
    vd_data = pd.read_csv("Video_Games_Sales.csv", dtype=vd_data_types)

    # Fill in missing year values with median
    yor_median = vd_data["Year_of_Release"].median()
    vd_data = vd_data.fillna(value={"Year_of_Release": yor_median})

    # Manually discretize scores; should be 10 bins ranging from 0 to 100
    score_bins = pdu.range_cut(0, 100, 10)

    def interval_map(x):
        i = 0
        for interval in score_bins:
            if x in interval:
                return i
            i = i + 1

    vd_data["Critic_Score"] = vd_data["Critic_Score"].map(interval_map).astype("category")

    # Train tree with video game data and test accuracy
    tree = DecisionTree(10)
    tree.fit(vd_data, limit=3)
    print(RenderTree(tree.root))
