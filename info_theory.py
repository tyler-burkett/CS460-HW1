import math
import types
import pandas as pd
import values_of from pandas_util

def entropy(samples):
    """
    Compute entropy of given set of sample data points

    Parameters:
    samples - Pandas DataFrame; last column is taken as the class labels
    """
    # Determine class label values
    class_labels = values_of(samples, samples.columns[-1])

    # Calculate entropy
    entropy_sum = 0
    for class_label in class_labels:
        # Create subset of samples by filtering items based on the class label
        if isinstance(class_label, types.FunctionType):
            samples_v = samples.apply(class_label, axis=1)
        else:
            samples_v = samples[samples[samples.columns[-1]].eq(class_label)]

        # Calculate probabilty of class label in samples and add to sum
        probabilty = len(samples_v)/len(samples)
        try:
            sum = sum + probabilty * math.log2(probabilty)
        except ValueError:
            sum = sum + 0
    return entropy_sum

def info_gain(samples, feature):
    """
    Compute information gain on set of samples if split based on
    provided feature.

    Parameters:
    samples - Pandas DataFrame; last column is taken as the class labels
    feature - Name of feature; Should correspond to column in samples
    """
    # Calculate weighted sum of subset entropies if samples split on feature
    values = values_of(samples, feature)

    # Calculate information gain
    entropy_sum = 0
    for value in values:
        # create subset of samples by filtering items based on the feature value
        if isinstance(value, types.FunctionType):
            samples_v = samples.apply(value, axis=1)
        else:
            samples_v = samples[samples[feature].eq(value)]

        # Calculate weighted entropy of subset and add to sum
        entropy_sum = entropy_sum + len(samples_v)/len(samples) * entropy(samples_v)

    return entropy(samples) - entropy_sum
