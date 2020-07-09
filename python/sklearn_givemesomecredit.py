import pandas as pd
from baseline.baseline import run_evaluation

train_samples = pd.read_csv('../datasets/givemesomecredit-train.csv', sep='\t')
test_samples = pd.read_csv('../datasets/givemesomecredit-test.csv', sep='\t')

label_attribute = 'label'
attribute_candidates = ['revolving_util','age','past_due','debt_ratio','income','lines','real_estate','dependents']

run_evaluation(train_samples, test_samples, attribute_candidates, label_attribute)
