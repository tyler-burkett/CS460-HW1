import pandas as pd
from pandas.core.dtypes.common import is_dtype_equal


def values_of(samples, feature, bins=None):
    """
    Determine possible values of feature for given.

    Parameters:
    samples - Pandas DataFrame; last column is taken as the class labels
    feature - Name of feature; should correspond to column in samples

    Keyword Args:
    bins - Number of bins/quantiles to have for continuous data; Default None
    """
    # If feature is already categorical, simply return the categories in use
    if is_dtype_equal(samples[feature].dtype, pd.api.types.CategoricalDtype):
        return samples[feature].dtype.categories

    # If feature is a string type, factorize the feature based on string value.
    # categories will be the different strings for the feature (i.e. assume these
    # are nominal values)
    if pd.api.types.is_string_dtype(samples[feature]):
        categories = pd.Categorical(samples[feature])
        return categories.dtype.categories

    # Default case. Assumed continuous data
    # Cut the feature values into bins
    if bins is None:
        raise ValueError("Missing required bin argument for continuous feature")
    return cut(samples[feature], bins)


def subset_by_value(samples, feature, value):
    """
    Subset samples based on value

    Parameters:
    samples - Pandas DataFrame; last column is taken as the class labels
    feature - Name of feature; should correspond to column in samples
    value - Value to check for subset membership; can be literal value or Index
    """
    if isinstance(value, pd.Interval):
        samples_v = samples[samples[feature].apply(lambda item: item in value)]
    else:
        samples_v = samples[samples[feature].eq(value)]
    return samples_v


def cut(values, bins, exterior_bins=False):
    """
    Create bins for continuous data. Returns Pandas Category object, with
    each category being an Interval (i.e. each bin is a range of values).
    This is meant to be behave somewhat like the cut method in Pandas.
    For simplicity, this method only creates equal sized bins.

    Parameters:
    values - Pandas Series; single column of DataFrame
    bins - Number of bins/quantiles to have for continuous data

    Keyword Args:
    exterior_bins - flag to include bins for values outside [min, max] range;
                    default True
    """
    # Calculate the size of each bin
    max = values.max(axis=0)
    min = values.min(axis=0)
    divisions = (max - min) / bins

    # Create the bins by defining them as a interval.
    # Construct in such a way that values contained in [min, max] belong to
    # exaclty one.
    intervals = []
    if(exterior_bins):
        intervals.append(pd.Interval(left=float("-inf"), right=min, closed="neither"))
    for i in range(bins):
        start = min + i * divisions
        end = min + (i + 1) * divisions if i != bins-1 else max
        if i == 0:
            intervals.append(pd.Interval(left=start, right=end, closed="both"))
        else:
            intervals.append(pd.Interval(left=start, right=end, closed="right"))
    if(exterior_bins):
        intervals.append(pd.Interval(left=max, right=float("inf"), closed="neither"))
    return intervals


def range_cut(min, max, bins, exterior_bins=False):
    """
    Wrapper function around cut() to specify min and max values directly. Useful
    if the range of expected values for a feature is know beforehand but not
    neccesarily present in a data set.

    Parameters:
    min - Minimum value of the range of values being discretized
    max - Maximum value of the range of values being discretized
    bins - Number of bins/quantiles to have for continuous data

    Keyword Args:
    exterior_bins - flag to include bins for values outside [min, max] range;
                    default True
    """
    values = pd.Series([min, max])
    return cut(values, bins, exterior_bins)
