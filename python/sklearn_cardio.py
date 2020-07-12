import pandas as pd
from experimentation.baseline import run_evaluation

train_samples = pd.read_csv('datasets/cardio-train.csv', sep='\t')
test_samples = pd.read_csv('datasets/cardio-test.csv', sep='\t')

label_attribute = 'label'
attribute_candidates = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'glucose', 'smoke',
                        'alcohol', 'active']

run_evaluation('cardio', train_samples, test_samples, attribute_candidates, label_attribute)
