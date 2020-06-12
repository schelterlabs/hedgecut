def binarize(row, attribute, positive_value):
    if row[attribute] == positive_value:
        return 1
    else:
        return 0