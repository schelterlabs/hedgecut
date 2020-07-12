import pandas as pd
from experimentation.baseline import run_evaluation

train_samples = pd.read_csv('datasets/shopping-train.csv', sep='\t')
test_samples = pd.read_csv('datasets/shopping-test.csv', sep='\t')

label_attribute = 'label'
attribute_candidates = ['administrative', 'administrative_duration', 'informational', 'informational_duration',
                        'product_related', 'product_related_duration', 'bounce_rates', 'exit_rates', 'page_values',
                        'special_day', 'month', 'operating_systems', 'browser', 'region', 'traffic_type',
                        'visitor_type', 'weekend']

run_evaluation('shopping', train_samples, test_samples, attribute_candidates, label_attribute)
