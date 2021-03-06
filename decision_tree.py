import pandas as pd
from anytree import AnyNode
from info_theory import info_gain
from pandas_util import values_of, subset_by_value


class DecisionTree:
    """Decsion Tree Using ID3 Algorithm"""

    def __init__(self, bins=2):
        self.bins = bins
        self.values = dict()

    def fit(self, training_data, limit=None):
        """
        Build a tree from the given training data.

        Paramters:
        training_data - Pandas DataFrame; last column is taken as the class labels

        Keyword Args:
        limit - Max depth limit of tree process; None by default
        """
        self.discretize_features(training_data)
        self.root = self.build_tree(training_data, limit)

    def discretize_features(self, data):
        """
        Method to build tree dictionary with possible values for each features
        and discretize as needed.

        Paramters:
        data - Pandas DataFrame; last column is taken as the class labels
        """

        # Extract possible values for each feature
        for column in data.columns[0:-1]:
            self.values[column] = values_of(data, column, self.bins)

    def predict(self, examples):
        """
        Predict given examples based on features.

        Paramters:
        examples - Pandas DataFrame; should have same columns as training data
        """
        results = []

        # Classify each example by traversing through the tree until you find
        # a leaf with a class label. Use that label as the classification
        for index, row in examples.iterrows():
            node = self.root
            while not hasattr(node, "label"):
                split_feature = node.attribute
                split_val = row[split_feature]
                found = False
                for child in node.children:
                    key = child.value
                    if (isinstance(key, pd.Interval) and split_val in key) or split_val == key:
                        node = child
                        found = True
                        break
                if(not found):
                    raise ValueError("Value out of range: Index {}".format(index))
            results.append(node.label)

        # Return results
        return pd.DataFrame(results, columns=["label"])

    def build_tree(self, training_data, limit=None):
        """
        Recursive function to build decision tree

        Parameters:
        training_data - Pandas DataFrame; last column is taken as the class labels

        Keyword Args:
        limit - Max depth limit of tree process; None by default
        """
        node = AnyNode()

        # Data is pure, create leaf of tree with class label
        if len(set(training_data.iloc[:, -1])) == 1:
            node.label = training_data.iloc[0, -1]
            return node

        # No more features to split on; use most common label as class label
        if len(training_data.columns) == 1:
            node.label = max(
                set(training_data.iloc[:, -1]),
                key=list(training_data.iloc[:, -1]).count)
            return node

        # Default; begin tree splitting
        # Determine feature that gives best information gain
        split_feature = max(training_data.columns[0:-1],
                            default=training_data.columns[0],
                            key=lambda x: info_gain(training_data, x, self.bins))
        node.attribute = split_feature

        # Lookup possible values for splitting feature and
        # create leaves/subtrees
        values = self.values[split_feature]
        for value in values:
            # Create subset with feature removed
            training_data_v = subset_by_value(training_data, split_feature, value)
            training_data_v = training_data_v.drop(split_feature, axis=1)

            # Subset data based on value
            if training_data_v.empty or (limit is not None and limit < 1):
                # subset is empty; create child leaf with label of the
                # most common class label
                child = AnyNode()
                child.label = max(
                    set(training_data.iloc[:, -1]),
                    key=list(training_data.iloc[:, -1]).count)
                child.value = value
            else:
                # subset is not empty; create child subtree recursively
                new_limit = None if limit is None else limit - 1
                child = self.build_tree(training_data_v, new_limit)
                child.value = value

            # Add new node as child of the current node and
            # map value to this child
            node.children = list(node.children) + [child]

        return node
