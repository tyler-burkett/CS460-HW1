import pandas as pd


def values_of(samples, feature, bins=None, equal_bins=None):
    """
    Determine possible values of feature for given. Create categories for

    Parameters:
    samples - Pandas DataFrame; last column is taken as the class labels
    feature - Name of feature; Should correspond to column in samples

    Keyword Args:
    bins - number of bins/quantiles to have for continuous data
    equal_bins - Boolean for discretizing based of frequency or not
                 (i.e bins will have roughly the same samples in each)
    """
    # If feature is already categorical, simply return the categories in use
    if pd.core.dtypes.common.is_dtype_equal(samples[feature].dtype,
                                            pd.api.types.CategoricalDtype):
        return samples[feature].dtype.categories

    # If feature is a string type, factorize the feature based on string value.
    # categories will be the different strings for the feature
    if pd.core.dtypes.common.is_dtype_equal(samples[feature].dtype, pd.api.types.StringDtype):
        coded_features, categories = pd.factorize(samples[feature])
        return categories

    # Default case. Assumed continuous data
    # Cut the feature values into bins according to
    if bins is None:
        raise ValueError('Missing required bin argument for continuous feature')
    if(equal_bins):
        return pd.qcut(samples[feature], bins)
    else:
        return pd.cut(samples[feature], bins)


def subset_by_value(samples, feature, value):
    """
    Subset samples based on value

    Parameters:
    samples - Pandas DataFrame; last column is taken as the class labels
    feature - Name of feature; Should correspond to column in samples
    value - Value to check for subset membership; can be literal value or Index
    """
    if isinstance(value, pd.Index):
        samples_v = samples[samples[feature] in value]
    else:
        samples_v = samples[samples[feature].eq(value)]
    return samples_v
