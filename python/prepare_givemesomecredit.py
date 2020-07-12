import pandas as pd

from experimentation.encoding import discretize, ordinalize, binarize

from sklearn.model_selection import train_test_split


df = pd.read_csv('datasets/givemesomecredit.csv', sep=',', na_values='NA')
df = df.dropna()

train_samples, test_samples = train_test_split(df, test_size=0.2)

revolving_util, revolving_util_discretizer = discretize(train_samples, 'RevolvingUtilizationOfUnsecuredLines')
age, age_discretizer = discretize(train_samples, 'age')
past_due, past_due_discretizer = discretize(train_samples, 'NumberOfTime30-59DaysPastDueNotWorse')
debt_ratio, debt_ratio_discretizer = discretize(train_samples, 'DebtRatio')
income, income_discretizer = discretize(train_samples, 'MonthlyIncome')
lines, lines_discretizer = discretize(train_samples, 'NumberOfOpenCreditLinesAndLoans')
real_estate, real_estate_discretizer = discretize(train_samples, 'NumberRealEstateLoansOrLines')
dependents, dependents_discretizer = discretize(train_samples, 'NumberOfDependents')
labels = train_samples['SeriousDlqin2yrs'].values


with open('datasets/givemesomecredit-train.csv', 'w') as file:

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
            str(int(real_estate[i][0])),
            str(int(dependents[i][0])),
            str(labels[i])
        ])
        file.write(line + '\n')

revolving_util = \
    revolving_util_discretizer.transform(test_samples['RevolvingUtilizationOfUnsecuredLines'].values.reshape(-1, 1))
age = age_discretizer.transform(test_samples['age'].values.reshape(-1, 1))
past_due = past_due_discretizer.transform(test_samples['NumberOfTime30-59DaysPastDueNotWorse'].values.reshape(-1, 1))
debt_ratio = debt_ratio_discretizer.transform(test_samples['DebtRatio'].values.reshape(-1, 1))
income = income_discretizer.transform(test_samples['MonthlyIncome'].values.reshape(-1, 1))
lines = lines_discretizer.transform(test_samples['NumberOfOpenCreditLinesAndLoans'].values.reshape(-1, 1))
real_estate = real_estate_discretizer.transform(test_samples['NumberRealEstateLoansOrLines'].values.reshape(-1, 1))
dependents = dependents_discretizer.transform(test_samples['NumberOfDependents'].values.reshape(-1, 1))
labels = test_samples['SeriousDlqin2yrs'].values

with open('datasets/givemesomecredit-test.csv', 'w') as file:

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
            str(int(real_estate[i][0])),
            str(int(dependents[i][0])),
            str(labels[i])
        ])
        file.write(line + '\n')