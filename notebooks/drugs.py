import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree, ensemble

from etd import ExtremelyRandomizedTrees

def binarize(row, attribute, positive_values):
    if row[attribute] in positive_values:
        return 1
    else:
        return 0

names = ['ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 
		 'Ascore', 'Cscore',  'Impulsive', 'SS', 'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 
		 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 
		 'Meth', 'Mushrooms', 'Nicotine', 'Semer','VSA']

samples = pd.read_csv('drugs.csv', names=names)



samples['label'] = samples.apply(lambda row: binarize(row, 'Ecstasy', ['CL3', 'CL4', 'CL5', 'CL6']), axis=1)

# print(len(samples))
# print(np.sum(samples['label']))

train_samples, test_samples = train_test_split(samples, test_size=0.2)

trees = ExtremelyRandomizedTrees(num_trees=10, d=5, n_min=3)

label_attribute = 'label'
attribute_candidates = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 
		 				'Ascore', 'Cscore',  'Impulsive', 'SS']

clf = trees.fit(train_samples, attribute_candidates, label_attribute)


y_pred = clf.predict(test_samples)
y_true = test_samples[label_attribute]

print('Accuracy (ETd)', accuracy_score(y_true, y_pred))
print('Confusion matrix (ETd)\n', confusion_matrix(y_true, y_pred))



X_train = train_samples[attribute_candidates].values
y_train = train_samples[label_attribute].values

clf_sklearn = tree.DecisionTreeClassifier()
clf_sklearn = clf_sklearn.fit(X_train, y_train)

etd_sklearn = ensemble.ExtraTreesClassifier()
etd_sklearn = etd_sklearn.fit(X_train, y_train)

X_test = test_samples[attribute_candidates].values
y_true_sklearn = test_samples[label_attribute].values
y_pred_sklearn = clf_sklearn.predict(X_test)

print('Accuracy (sklearn)', accuracy_score(y_true_sklearn, y_pred_sklearn))
print('Confusion matrix (sklearn)\n', confusion_matrix(y_true_sklearn, y_pred_sklearn))

y_pred_sklearn_etd = etd_sklearn.predict(X_test)

print('Accuracy (ETD sklearn)', accuracy_score(y_true_sklearn, y_pred_sklearn_etd))
print('Confusion matrix (ETD sklearn)\n', confusion_matrix(y_true_sklearn, y_pred_sklearn_etd))