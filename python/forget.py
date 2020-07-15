import pandas as pd
from experimentation.baseline import forget

label_attribute = 'label'

train_samples = pd.read_csv('datasets/adult-train.csv', sep='\t')
attribute_candidates = ['age', 'workclass', 'fnlwgt', 'education', 'marital_status', 'occupation', 'relationship',
                        'race', 'sex', 'capital_gain', 'hours_per_week', 'native_country']

forget('adult', train_samples, attribute_candidates, label_attribute)


train_samples = pd.read_csv('datasets/cardio-train.csv', sep='\t')
attribute_candidates = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'glucose', 'smoke',
                        'alcohol', 'active']

forget('cardio', train_samples, attribute_candidates, label_attribute)


train_samples = pd.read_csv('datasets/givemesomecredit-train.csv', sep='\t')
attribute_candidates = ['revolving_util', 'age', 'past_due', 'debt_ratio', 'income', 'lines',
                        'real_estate', 'dependents']

forget('givemesomecredit', train_samples, attribute_candidates, label_attribute)


train_samples = pd.read_csv('datasets/propublica-train.csv', sep='\t')
attribute_candidates = ['age', 'decile_score', 'priors_count', 'days_b_screening_arrest', 'is_recid',
                        'c_charge_degree', 'sex', 'age_cat', 'score_text', 'race']

forget('propublica', train_samples, attribute_candidates, label_attribute)


train_samples = pd.read_csv('datasets/shopping-train.csv', sep='\t')
attribute_candidates = ['administrative', 'administrative_duration', 'informational', 'informational_duration',
                        'product_related', 'product_related_duration', 'bounce_rates', 'exit_rates', 'page_values',
                        'special_day', 'month', 'operating_systems', 'browser', 'region', 'traffic_type',
                        'visitor_type', 'weekend']

forget('shopping', train_samples, attribute_candidates, label_attribute)
