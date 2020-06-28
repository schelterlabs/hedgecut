import numpy as np
import pandas as pd
import duckdb

from sklearn.preprocessing import KBinsDiscretizer


def binarize(row, attribute, positive_value):
    if row[attribute] == positive_value:
        return 1
    else:
        return 0


def discretize(data, attribute):
	discretizer = KBinsDiscretizer(n_bins=16, encode='ordinal', strategy='quantile')
	return discretizer.fit_transform(data[attribute].values.reshape(-1, 1))

conn = duckdb.connect()

conn.execute("CREATE TABLE titanic_raw AS SELECT * FROM read_csv_auto('../titanic.csv');")

raw_data = conn.execute("SELECT * FROM titanic_raw").fetchdf()


ages = discretize(raw_data, 'Age')
fares = discretize(raw_data, 'Fare')
siblings = discretize(raw_data, 'Siblings/Spouses Aboard')
children = discretize(raw_data, 'Parents/Children Aboard')
genders = raw_data.apply(lambda row: binarize(row, 'Sex', 'male'), axis=1)
pclasses = raw_data['Pclass'].values
labels = raw_data['Survived'].values

record_id = 0

with open('../titanic-attributes.csv', 'w') as attributes_file:

	attributes_file.write(f'record_id\tattribute\tvalue\n')

	for (age, fare, gender, pclass, label) in zip(ages, fares, genders, pclasses, labels):

		attributes_file.write(f'{record_id}\tage\t{int(age[0])}\n');
		attributes_file.write(f'{record_id}\tfare\t{int(fare[0])}\n');
		attributes_file.write(f'{record_id}\tsiblings\t{int(siblings[0])}\n')
		attributes_file.write(f'{record_id}\tchildren\t{int(children[0])}\n')
		attributes_file.write(f'{record_id}\tgender\t{int(gender)}\n')
		attributes_file.write(f'{record_id}\tpclass\t{int(pclass)}\n')
		attributes_file.write(f'{record_id}\tlabel\t{label}\n')

		record_id += 1