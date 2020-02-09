import pandas as pd
from anytree import AnyNode
from info_theory import info_gain
from pandas_util import values_of, subset_by_value


class DecisionTree:
    """Decsion Tree Using ID3 Algorithm"""

    def __init__(self, bins=2):
        self.bins = bins

    def fit(self, training_data, limit=None):
        self.root = self.build_tree(training_data, limit)

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
            while not hasattr(node, 'label'):
                split_feature = node.attribute
                split_val = row[split_feature]
                for key, child in node.map:
                    if (isinstance(key, pd.Interval) and split_val in key) or split_val == key:
                        node = child
                        break
            results.append(node.label)

        # Return results
        return pd.DataFrame(results, columns=['label'])

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
            node.label = training_data.iloc[1, -1]
            return node

        # No more features to split on; use most common label as class label
        if len(training_data.columns == 1):
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

        # Determine possible values for splitting feature and
        # create leaves/subtrees
        values = values_of(training_data, split_feature, self.bins)
        node.map = []
        for value in values:
            # Create subset with feature removed
            training_data_v = subset_by_value(training_data, split_feature, value)
            training_data_v.drop(split_feature, axis=1)

            # Subset data based on value
            if training_data_v.empty or limit < 1 or limit is None:
                # subset is empty; create child leaf with label of the
                # most common class label
                child = AnyNode()
                child.label = max(
                    set(training_data.iloc[:, -1]),
                    key=list(training_data.iloc[:, -1]).count)
            else:
                # subset is not empty; create child subtree recursively
                new_limit = limit - 1
                child = self.build_tree(training_data_v, new_limit)

            # Add new node as child of the current node and
            # map value to this child
            node.children = list(node.children) + [child]
            node.map.append((value, child))

        return node
