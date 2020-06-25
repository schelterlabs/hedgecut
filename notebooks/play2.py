import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree

from utils import binarize
from etd import ExtremelyRandomizedTrees


samples = pd.read_csv('../occupancy.csv')
#samples['sex_binary'] = samples.apply(lambda row: binarize(row, 'Sex', 'male'), axis=1)


train_samples, test_samples = train_test_split(samples, test_size=0.2)

trees = ExtremelyRandomizedTrees(num_trees=1, d=5, n_min=3)

label_attribute = 'Occupancy'
attribute_candidates = ["Humidity", "Light", "CO2", "HumidityRatio"]


# 1 really bad case
# 5 bad case
# 8,14 minor case

np.random.seed(1)
clf = trees.fit(train_samples, attribute_candidates, label_attribute)

for row in [88]:#range(1, 100):

	print("Testing forgetting of", row)
	sample_to_forget = train_samples.iloc[row]
	label_to_forget = sample_to_forget[label_attribute]

	clf.forget(sample_to_forget, label_to_forget)


# print(clf.trees[0].attribute, clf.trees[0].cut_off)
# print('\t', clf.trees[0].left_node.attribute, clf.trees[0].left_node.cut_off)
# #print("\t\t", clf.trees[0].left_node.left_node.attribute, clf.trees[0].left_node.left_node.cut_off)
# print("\t\t", clf.trees[0].left_node.right_node.attribute, clf.trees[0].left_node.right_node.cut_off)
# print('\t', clf.trees[0].right_node.attribute, clf.trees[0].right_node.cut_off)
# print("\t\t", clf.trees[0].right_node.left_node.attribute, clf.trees[0].right_node.left_node.cut_off)
# print("\t\t", clf.trees[0].right_node.right_node.attribute, clf.trees[0].right_node.right_node.cut_off)

# np.random.seed(42)

# one_out = train_samples.iloc[10:]

# clf2 = trees.fit(one_out, attribute_candidates, label_attribute)

# print(clf2.trees[0].attribute, clf2.trees[0].cut_off)
# print('\t', clf2.trees[0].left_node.attribute, clf2.trees[0].left_node.cut_off)
# #print("\t\t", clf2.trees[0].left_node.left_node.attribute, clf2.trees[0].left_node.left_node.cut_off)
# print("\t\t", clf2.trees[0].left_node.right_node.attribute, clf2.trees[0].left_node.right_node.cut_off)
# print('\t', clf2.trees[0].right_node.attribute, clf2.trees[0].right_node.cut_off)
# print("\t\t", clf2.trees[0].right_node.left_node.attribute, clf2.trees[0].right_node.left_node.cut_off)
# print("\t\t", clf2.trees[0].right_node.right_node.attribute, clf2.trees[0].right_node.right_node.cut_off)