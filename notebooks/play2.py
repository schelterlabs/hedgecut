from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree

from utils import binarize
from etd import ExtremelyRandomizedTrees

seed = 14

samples = pd.read_csv('../occupancy.csv')
#samples['sex_binary'] = samples.apply(lambda row: binarize(row, 'Sex', 'male'), axis=1)



train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=seed)

trees = ExtremelyRandomizedTrees(num_trees=1, d=5, n_min=3)

label_attribute = 'Occupancy'
attribute_candidates = ["Humidity", "Light", "CO2", "HumidityRatio"]


# 1 really bad case
# 5 bad case
# 8,14 minor case

np.random.seed(seed)

clf = trees.fit(train_samples, attribute_candidates, label_attribute)


np.random.seed(123456)
for _ in range(1, 25):

	clf_clone = deepcopy(clf)

	rows_to_remove = np.random.choice(len(train_samples), 6)

	print(rows_to_remove)

	for row in rows_to_remove:
		#print("Testing forgetting of", row)
		sample_to_forget = train_samples.iloc[row]
		label_to_forget = sample_to_forget[label_attribute]

		clf.forget(sample_to_forget, label_to_forget)

