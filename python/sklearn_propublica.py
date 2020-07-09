import pandas as pd
from baseline.baseline import run_evaluation

train_samples = pd.read_csv('../datasets/propublica-train.csv', sep='\t')
test_samples = pd.read_csv('../datasets/propublica-test.csv', sep='\t')

label_attribute = 'label'
attribute_candidates = ['age', 'decile_score', 'priors_count', 'days_b_screening_arrest', 'is_recid',
                        'c_charge_degree', 'sex', 'age_cat', 'score_text', 'race']

run_evaluation(train_samples, test_samples, attribute_candidates, label_attribute)
