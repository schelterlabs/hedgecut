import pandas as pd
from baseline.baseline import run_evaluation

train_samples = pd.read_csv('../datasets/defaults-train.csv', sep='\t')
test_samples = pd.read_csv('../datasets/defaults-test.csv', sep='\t')

label_attribute = 'label'
attribute_candidates = ['limit', 'sex', 'education', 'marriage', 'age', 'pay0', 'pay2', 'pay3', 'pay4', 'pay5',
                        'pay6', 'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
                        'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6']

run_evaluation(train_samples, test_samples, attribute_candidates, label_attribute)
