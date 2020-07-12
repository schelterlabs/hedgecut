
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder


def discretize(data, attribute):
    discretizer = KBinsDiscretizer(n_bins=16, encode='ordinal', strategy='quantile')
    discretizer = discretizer.fit(data[attribute].values.reshape(-1, 1))
    transformed_values = discretizer.transform(data[attribute].values.reshape(-1, 1))
    return transformed_values, discretizer


def ordinalize(all_data, data, attribute):
    encoder = LabelEncoder()
    encoder = encoder.fit(all_data[attribute].values.reshape(1, -1)[0])
    transformed_values = encoder.transform(data[attribute].values.reshape(1, -1)[0])
    return transformed_values, encoder


def binarize(row, attribute, positive_value):
    if row[attribute] == positive_value:
        return 1
    else:
        return 0
