import numpy as np
from anytree import AnyNode

class DecisionTree:
    """Decsion Tree Using ID3 Algorithm"""

    def __init__(self):
        pass

    def fit(self, training_data, limit=None):
        self.root = build_tree(training_data, limit)

    def predict(self, examples):
        pass

    def build_tree(self, training_data, limit=None):
        """Recursive function to build decision tree

        Parameters:
        training_data - Pandas Dataframe; last column is taken as the class class_labels

        Keyword Args:
        limit - Max depth limit of tree process; None by default
        """
        node = AnyNode()

        # Data is pure, create leaf of tree with class label
        if len(set(training_data.iloc[:,-1])) == 1:
            node.label = training_data.iloc[1,-1]
            return node

        # No more attributes to split on; use most common label as class label
        if len(training_data.columns == 1):
            node.label = max(set(training_data.iloc[:,-1]), key = list(training_data.iloc[:,-1]).count)
            return node

        # Begin splitting on attribute
