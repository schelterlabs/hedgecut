import numpy as np
import pandas as pd
import duckdb

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

def binarize(row, attribute, positive_value):
    if row[attribute] == positive_value:
        return 1
    else:
        return 0


def discretize(data, attribute):
    discretizer = KBinsDiscretizer(n_bins=16, encode='ordinal', strategy='quantile')
    discretizer = discretizer.fit(data[attribute].values.reshape(-1, 1))
    transformed_values = discretizer.transform(data[attribute].values.reshape(-1, 1))
    return transformed_values, discretizer

conn = duckdb.connect()

conn.execute("CREATE TABLE titanic_raw AS SELECT * FROM read_csv_auto('../datasets/titanic.csv');")

raw_data = conn.execute("SELECT * FROM titanic_raw").fetchdf()

train_samples, test_samples = train_test_split(raw_data, test_size=0.2)

ages, age_discretizer = discretize(train_samples, 'Age')
fares, fares_discretizer = discretize(train_samples, 'Fare')
siblings = train_samples['Siblings/Spouses Aboard'].values
children = train_samples['Parents/Children Aboard'].values
genders = train_samples.apply(lambda row: binarize(row, 'Sex', 'male'), axis=1)
pclasses = train_samples['Pclass'].values - 1
labels = train_samples['Survived'].values

record_id = 0

with open('../datasets/titanic-train.csv', 'w') as file:

    file.write(f'record_id\tage\tfare\tsiblings\tchildren\tgender\tpclass\tlabel\n')

    for (age, fare, sibling, thechildren, gender, pclass, label) in zip(ages, fares, siblings, children, genders, pclasses, labels):

        file.write(f'{record_id}\t{int(age[0])}\t{int(fare[0])}\t{int(sibling)}\t{int(thechildren)}\t{int(gender)}\t{int(pclass)}\t{label}\n')
        record_id += 1

test_ages = age_discretizer.transform(test_samples['Age'].values.reshape(-1, 1))
test_fares = fares_discretizer.transform(test_samples['Fare'].values.reshape(-1, 1))
test_siblings = test_samples['Siblings/Spouses Aboard'].values
test_children = test_samples['Parents/Children Aboard'].values
test_genders = test_samples.apply(lambda row: binarize(row, 'Sex', 'male'), axis=1)
test_pclasses = test_samples['Pclass'].values - 1
test_labels = test_samples['Survived'].values

with open('../datasets/titanic-test.csv', 'w') as file:

    file.write(f'record_id\tage\tfare\tsiblings\tchildren\tgender\tpclass\tlabel\n')

    for (age, fare, sibling, thechildren, gender, pclass, label) in zip(test_ages, test_fares, test_siblings, test_children, test_genders, test_pclasses, test_labels):

        file.write(f'{record_id}\t{int(age[0])}\t{int(fare[0])}\t{int(sibling)}\t{int(thechildren)}\t{int(gender)}\t{int(pclass)}\t{label}\n')
        record_id += 1
