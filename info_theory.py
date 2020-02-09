import math
from pandas_util import values_of, subset_by_value


def entropy(samples, bins=None):
    """
    Compute entropy of given set of sample data points

    Parameters:
    samples - Pandas DataFrame; last column is taken as the class labels

    Keyword Args:
    bins - number of bins/quantiles to have for continuous data
    """
    # Determine class label values
    class_labels = values_of(samples, samples.columns[-1], bins)

    # Calculate entropy
    entropy_sum = 0
    for class_label in class_labels:
        # Create subset of samples by filtering items based on the class label
        samples_v = subset_by_value(samples, samples.columns[-1], class_label)

        # Calculate part of entropy sum for class label
        probabilty = len(samples_v)/len(samples)
        try:
            entropy_sum = entropy_sum + probabilty * math.log2(probabilty)
        except ValueError:
            entropy_sum = entropy_sum + 0
    return entropy_sum


def info_gain(samples, feature, bins=None):
    """
    Compute information gain on set of samples if split based on
    provided feature.

    Parameters:
    samples - Pandas DataFrame; last column is taken as the class labels
    feature - Name of feature; should correspond to column in samples

    Keyword Args:
    bins - number of bins/quantiles to have for continuous data
    """
    # Determine possible values of feature
    values = values_of(samples, feature)

    # Calculate information gain
    entropy_sum = 0
    for value in values:
        # Create samples subset by filtering items based on the feature value
        samples_v = subset_by_value(samples, feature, value)

        # Calculate weighted entropy of subset and add to sum
        entropy_sum = entropy_sum + \
            len(samples_v)/len(samples) * entropy(samples_v)

    return entropy(samples) - entropy_sum
