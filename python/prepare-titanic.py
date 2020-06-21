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
labels = raw_data['Survived'].values

record_id = 0

with open('titanic-attributes.csv', 'w') as attributes_file:
	with open('titanic-labels.csv', 'w') as labels_file:

		attributes_file.write(f'record_id\tattribute\tvalue\n')
		labels_file.write(f'record_id\tlabel\n')

		for (age, fare, gender, label) in zip(ages, fares, genders, labels):

			attributes_file.write(f'{record_id}\tage\t{int(age[0])}\n');
			attributes_file.write(f'{record_id}\tfare\t{int(fare[0])}\n');
			attributes_file.write(f'{record_id}\tsiblings\t{int(siblings[0])}\n')
			attributes_file.write(f'{record_id}\tchildren\t{int(children[0])}\n')
			attributes_file.write(f'{record_id}\tgender\t{int(gender)}\n')

			labels_file.write(f'{record_id}\t{label}\n')
		#print(record_id, int(age[0]), int(fare[0]), int(siblings[0]), int(children[0]), int(gender), label)	    
			record_id += 1