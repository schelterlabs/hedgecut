import time
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree, ensemble


def train_time(name, train_samples, attribute_candidates, label_attribute):
    X_train = train_samples[attribute_candidates].values
    y_train = train_samples[label_attribute].values

    t = time.process_time()
    clf_sklearn = tree.DecisionTreeClassifier()
    clf_sklearn.fit(X_train, y_train)
    dt_train_time = time.process_time() - t

    t = time.process_time()
    clf_sklearn_rf = ensemble.RandomForestClassifier()
    clf_sklearn_rf.fit(X_train, y_train)
    rf_train_time = time.process_time() - t

    t = time.process_time()
    etd_sklearn = ensemble.ExtraTreesClassifier(n_estimators=100,
                                                criterion='gini',
                                                min_samples_leaf=2,
                                                max_features='sqrt')
    etd_sklearn.fit(X_train, y_train)
    etd_train_time = time.process_time() - t

    print(f'{name},decision_tree,{int(dt_train_time * 1000)}')
    print(f'{name},random_forest,{int(rf_train_time * 1000)}')
    print(f'{name},extremely_randomized_trees,{int(etd_train_time * 1000)}')


def forget(name, train_samples, attribute_candidates, label_attribute):

    train_samples = train_samples.sample(frac=1).reset_index(drop=True)

    repetitions = int(len(train_samples) / 1000)

    for _ in range(0, repetitions):
        train_samples = train_samples.iloc[1:]

        X_train = train_samples[attribute_candidates].values
        y_train = train_samples[label_attribute].values

        t = time.process_time()
        clf_sklearn = tree.DecisionTreeClassifier()
        clf_sklearn.fit(X_train, y_train)
        dt_train_time = time.process_time() - t

        t = time.process_time()
        clf_sklearn_rf = ensemble.RandomForestClassifier()
        clf_sklearn_rf.fit(X_train, y_train)
        rf_train_time = time.process_time() - t

        t = time.process_time()
        etd_sklearn = ensemble.ExtraTreesClassifier(n_estimators=100,
                                                    criterion='gini',
                                                    min_samples_leaf=2,
                                                    max_features='sqrt')
        etd_sklearn.fit(X_train, y_train)
        etd_train_time = time.process_time() - t

        print(f'{name},decision_tree,{int(dt_train_time * 1000000)}')
        print(f'{name},random_forest,{int(rf_train_time * 1000000)}')
        print(f'{name},extremely_randomized_trees,{int(etd_train_time * 1000000)}')


def run_evaluation(name, train_samples, test_samples, attribute_candidates, label_attribute):
    X_train = train_samples[attribute_candidates].values
    y_train = train_samples[label_attribute].values

    t = time.process_time()
    clf_sklearn = tree.DecisionTreeClassifier()
    clf_sklearn = clf_sklearn.fit(X_train, y_train)
    dt_train_time = time.process_time() - t

    t = time.process_time()
    clf_sklearn_rf = ensemble.RandomForestClassifier()
    clf_sklearn_rf = clf_sklearn_rf.fit(X_train, y_train)
    rf_train_time = time.process_time() - t

    t = time.process_time()
    etd_sklearn = ensemble.ExtraTreesClassifier(n_estimators=100,
                                                criterion='gini',
                                                min_samples_leaf=2,
                                                max_features='sqrt',
                                                n_jobs=-1)
    etd_sklearn = etd_sklearn.fit(X_train, y_train)
    etd_train_time = time.process_time() - t

    t = time.process_time()
    etd_sklearn_single = ensemble.ExtraTreesClassifier(n_estimators=100,
                                                       criterion='gini',
                                                       min_samples_leaf=2,
                                                       max_features='sqrt')
    etd_sklearn_single = etd_sklearn_single.fit(X_train, y_train)
    etd_train_time_single = time.process_time() - t

    X_test = test_samples[attribute_candidates].values
    y_true_sklearn = test_samples[label_attribute].values
    y_pred_sklearn = clf_sklearn.predict(X_test)

    #print('Train time (sklearn)', dt_train_time)
    #print('Accuracy (sklearn)', accuracy_score(y_true_sklearn, y_pred_sklearn))
    #print('Confusion matrix (sklearn)\n', confusion_matrix(y_true_sklearn, y_pred_sklearn))

    y_pred_sklearn_rf = clf_sklearn_rf.predict(X_test)

    #print('Train time (RF sklearn)', rf_train_time)
    #print('Accuracy (RF sklearn)', accuracy_score(y_true_sklearn, y_pred_sklearn_rf))
    #print('Confusion matrix (sklearn)\n', confusion_matrix(y_true_sklearn, y_pred_sklearn_rf))

    y_pred_sklearn_etd = etd_sklearn.predict(X_test)

    #print('Train time (ETD sklearn)', etd_train_time)
    #print('Accuracy (ETD sklearn)', accuracy_score(y_true_sklearn, y_pred_sklearn_etd))
    #print('Confusion matrix (ETD sklearn)\n', confusion_matrix(y_true_sklearn, y_pred_sklearn_etd))

    y_pred_sklearn_etd_single = etd_sklearn_single.predict(X_test)

    #print('Train time (ETD sklearn single)', etd_train_time_single)
    #print('Accuracy (ETD sklearn)', accuracy_score(y_true_sklearn, y_pred_sklearn_etd_single))
    #print('Confusion matrix (ETD sklearn)\n', confusion_matrix(y_true_sklearn, y_pred_sklearn_etd_single))

    print(f'{name},decision_tree,{accuracy_score(y_true_sklearn, y_pred_sklearn)}')
    print(f'{name},random_forest,{accuracy_score(y_true_sklearn, y_pred_sklearn_rf)}')
    print(f'{name},extremely_randomized_trees,{accuracy_score(y_true_sklearn, y_pred_sklearn_etd_single)}')