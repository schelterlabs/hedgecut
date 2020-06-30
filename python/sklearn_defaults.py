import time
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree, ensemble

train_samples = pd.read_csv('../datasets/defaults-train.csv', sep='\t')
test_samples = pd.read_csv('../datasets/defaults-test.csv', sep='\t')

label_attribute = 'label'
attribute_candidates = ['limit', 'sex', 'education', 'marriage', 'age', 'pay0', 'pay2', 'pay3', 'pay4', 'pay5',
                        'pay6', 'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
                        'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6']

X_train = train_samples[attribute_candidates].values
y_train = train_samples[label_attribute].values


t = time.process_time()
clf_sklearn = tree.DecisionTreeClassifier()
clf_sklearn = clf_sklearn.fit(X_train, y_train)
dt_train_time = time.process_time() - t

# TODO add variant with single thread (and figure out why the heck it is so fast...)
t = time.process_time()
etd_sklearn = ensemble.ExtraTreesClassifier(n_estimators=100,
                                            criterion='entropy',
                                            min_samples_leaf=2,
                                            max_features='sqrt',
                                            n_jobs=-1)
etd_sklearn = etd_sklearn.fit(X_train, y_train)
etd_train_time = time.process_time() - t


X_test = test_samples[attribute_candidates].values
y_true_sklearn = test_samples[label_attribute].values
y_pred_sklearn = clf_sklearn.predict(X_test)

print('Train time (sklearn)', dt_train_time)
print('Accuracy (sklearn)', accuracy_score(y_true_sklearn, y_pred_sklearn))
print('Confusion matrix (sklearn)\n', confusion_matrix(y_true_sklearn, y_pred_sklearn))

y_pred_sklearn_etd = etd_sklearn.predict(X_test)

print('Train time (ETD sklearn)', etd_train_time)
print('Accuracy (ETD sklearn)', accuracy_score(y_true_sklearn, y_pred_sklearn_etd))
print('Confusion matrix (ETD sklearn)\n', confusion_matrix(y_true_sklearn, y_pred_sklearn_etd))