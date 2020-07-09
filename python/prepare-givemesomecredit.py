import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.model_selection import train_test_split


def discretize(data, attribute):
    discretizer = KBinsDiscretizer(n_bins=16, encode='ordinal', strategy='quantile')
    discretizer = discretizer.fit(data[attribute].values.reshape(-1, 1))
    transformed_values = discretizer.transform(data[attribute].values.reshape(-1, 1))
    return transformed_values, discretizer


def ordinalize(data, attribute):
    encoder = LabelEncoder()
    encoder = encoder.fit(data[attribute].values.reshape(1, -1)[0])
    transformed_values = encoder.transform(data[attribute].values.reshape(1, -1)[0])
    return transformed_values, encoder


def binarize(row, attribute, positive_value):
    if str(row[attribute]) == positive_value:
        return 1
    else:
        return 0


df = pd.read_csv('../datasets/givemesomecredit.csv', sep=',', na_values='NA')
df = df.dropna()

train_samples, test_samples = train_test_split(df, test_size=0.2)

# 'RevolvingUtilizationOfUnsecuredLines','age','NumberOfTime30-59DaysPastDueNotWorse',
# 'DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate',
# 'NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents']

revolving_util, revolving_util_discretizer = discretize(train_samples, 'RevolvingUtilizationOfUnsecuredLines')
age, age_discretizer = discretize(train_samples, 'age')
past_due, past_due_discretizer = discretize(train_samples, 'NumberOfTime30-59DaysPastDueNotWorse')
debt_ratio, debt_ratio_discretizer = discretize(train_samples, 'DebtRatio')
income, income_discretizer = discretize(train_samples, 'MonthlyIncome')
lines, lines_discretizer = discretize(train_samples, 'NumberOfOpenCreditLinesAndLoans')
#late, late_discretizer = discretize(train_samples, 'NumberOfTimes90DaysLate')
real_estate, real_estate_discretizer = discretize(train_samples, 'NumberRealEstateLoansOrLines')
#past_due_days, past_due_days_discretizer = discretize(train_samples, 'NumberOfTime60-89DaysPastDueNotWorse')
dependents, dependents_discretizer = discretize(train_samples, 'NumberOfDependents')
labels = train_samples['SeriousDlqin2yrs'].values

print(np.max(revolving_util))
print(np.max(age))
print(np.max(past_due))
print(np.max(debt_ratio))
print(np.max(income))
print(np.max(lines))
#print(np.max(late))
print(np.max(real_estate))
#print(np.max(past_due_days))
print(np.max(dependents))

with open('../datasets/givemesomecredit-train.csv', 'w') as file:

    file.write(f'record_id\trevolving_util\tage\tpast_due\tdebt_ratio\tincome\tlines\treal_estate\t' +
               f'dependents\tlabel\n')

    for i in range(0, len(train_samples)):
        line = '\t'.join([
            str(i),
            str(int(revolving_util[i][0])),
            str(int(age[i][0])),
            str(int(past_due[i][0])),
            str(int(debt_ratio[i][0])),
            str(int(income[i][0])),
            str(int(lines[i][0])),
            #str(int(late[i][0])),
            str(int(real_estate[i][0])),
            #str(int(past_due_days[i][0])),
            str(int(dependents[i][0])),
            str(labels[i])
        ])
        file.write(line + '\n')

revolving_util = revolving_util_discretizer.transform(test_samples['RevolvingUtilizationOfUnsecuredLines'].values.reshape(-1, 1))
age = age_discretizer.transform(test_samples['age'].values.reshape(-1, 1))
past_due = past_due_discretizer.transform(test_samples['NumberOfTime30-59DaysPastDueNotWorse'].values.reshape(-1, 1))
debt_ratio = debt_ratio_discretizer.transform(test_samples['DebtRatio'].values.reshape(-1, 1))
income = income_discretizer.transform(test_samples['MonthlyIncome'].values.reshape(-1, 1))
lines = lines_discretizer.transform(test_samples['NumberOfOpenCreditLinesAndLoans'].values.reshape(-1, 1))
#late = late_discretizer.transform(test_samples['NumberOfTimes90DaysLate'].values.reshape(-1, 1))
real_estate = real_estate_discretizer.transform(test_samples['NumberRealEstateLoansOrLines'].values.reshape(-1, 1))
#past_due_days = past_due_days_discretizer.transform(test_samples['NumberOfTime60-89DaysPastDueNotWorse'].values.reshape(-1, 1))
dependents = dependents_discretizer.transform(test_samples['NumberOfDependents'].values.reshape(-1, 1))
labels = test_samples['SeriousDlqin2yrs'].values

with open('../datasets/givemesomecredit-test.csv', 'w') as file:

    file.write(f'record_id\trevolving_util\tage\tpast_due\tdebt_ratio\tincome\tlines\treal_estate\t' +
               f'dependents\tlabel\n')

    for i in range(0, len(test_samples)):
        line = '\t'.join([
            str(i + len(train_samples)),
            str(int(revolving_util[i][0])),
            str(int(age[i][0])),
            str(int(past_due[i][0])),
            str(int(debt_ratio[i][0])),
            str(int(income[i][0])),
            str(int(lines[i][0])),
            #str(int(late[i][0])),
            str(int(real_estate[i][0])),
            #str(int(past_due_days[i][0])),
            str(int(dependents[i][0])),
            str(labels[i])
        ])
        file.write(line + '\n')