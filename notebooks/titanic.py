import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree, ensemble

from utils import binarize
from etd import ExtremelyRandomizedTrees



samples = pd.read_csv('../titanic.csv')
samples['sex_binary'] = samples.apply(lambda row: binarize(row, 'Sex', 'male'), axis=1)


train_samples, test_samples = train_test_split(samples, test_size=0.2)

trees = ExtremelyRandomizedTrees(num_trees=10, d=5, n_min=3)

label_attribute = 'Survived'
attribute_candidates = ['Age', 'Pclass', 'Fare', 'Siblings/Spouses Aboard',
						'Parents/Children Aboard', 'sex_binary']


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