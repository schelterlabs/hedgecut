import pandas as pd
from experimentation.baseline import run_evaluation

train_samples = pd.read_csv('datasets/adult-train.csv', sep='\t')
test_samples = pd.read_csv('datasets/adult-test.csv', sep='\t')

label_attribute = 'label'
attribute_candidates = ['age', 'workclass', 'fnlwgt', 'education', 'marital_status', 'occupation', 'relationship',
                        'race', 'sex', 'capital_gain', 'hours_per_week', 'native_country']

run_evaluation('adult', train_samples, test_samples, attribute_candidates, label_attribute)
