import pandas as pd

from experimentation.encoding import discretize, ordinalize, binarize

from sklearn.model_selection import train_test_split


names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
         'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

raw_data = pd.read_csv('datasets/adult.csv', sep=', ', na_values='?', names=names, index_col=False, engine='python')
raw_data = raw_data.dropna()

train_samples, test_samples = train_test_split(raw_data, test_size=0.2)

age, age_discretizer = discretize(train_samples, 'age')
workclass, workclass_encoder = ordinalize(raw_data, train_samples, 'workclass')
fnlwgt, fnlwgt_discretizer = discretize(train_samples, 'fnlwgt')
education, education_encoder = ordinalize(raw_data, train_samples, 'education')
marital_status, marital_status_encoder = ordinalize(raw_data, train_samples, 'marital-status')
occupation, occupation_encoder = ordinalize(raw_data, train_samples, 'occupation')
relationship, relationship_encoder = ordinalize(raw_data, train_samples, 'relationship')
race, race_encoder = ordinalize(raw_data, train_samples, 'race')
sex, sex_encoder = ordinalize(raw_data, train_samples, 'sex')
capital_gain, capital_gain_discretizer = discretize(train_samples, 'capital-gain')
capital_loss, capital_loss_discretizer = discretize(train_samples, 'capital-loss')
hours_per_week, hours_per_week_discretizer = discretize(train_samples, 'hours-per-week')
native_country, native_country_encoder = ordinalize(raw_data, train_samples, 'native-country')
labels = train_samples.apply(lambda row: binarize(row, 'income', '>50K'), axis=1).values


with open('datasets/adult-train.csv', 'w') as file:

    file.write(f'record_id\tage\tworkclass\tfnlwgt\teducation\tmarital_status\toccupation\trelationship\trace\tsex'
               f'\tcapital_gain\thours_per_week\tnative_country\tlabel\n')

    for i in range(0, len(train_samples)):
        line = '\t'.join([
            str(i),
            str(int(age[i][0])),
            str(workclass[i]),
            str(int(fnlwgt[i][0])),
            str(education[i]),
            str(marital_status[i]),
            str(occupation[i]),
            str(relationship[i]),
            str(race[i]),
            str(sex[i]),
            str(int(capital_gain[i][0])),
            str(int(hours_per_week[i][0])),
            str(native_country[i]),
            str(labels[i])
        ])
        file.write(line + '\n')

age = age_discretizer.transform(test_samples['age'].values.reshape(-1, 1))
workclass = workclass_encoder.transform(test_samples['workclass'].values.reshape(-1, 1))
fnlwgt = fnlwgt_discretizer.transform(test_samples['fnlwgt'].values.reshape(-1, 1))
education = education_encoder.transform(test_samples['education'].values.reshape(-1, 1))
marital_status = marital_status_encoder.transform(test_samples['marital-status'].values.reshape(-1, 1))
occupation = occupation_encoder.transform(test_samples['occupation'].values.reshape(-1, 1))
relationship = relationship_encoder.transform(test_samples['relationship'].values.reshape(-1, 1))
race = race_encoder.transform(test_samples['race'].values.reshape(-1, 1))
sex = sex_encoder.transform(test_samples['sex'].values.reshape(-1, 1))
capital_gain = capital_gain_discretizer.transform(test_samples['capital-gain'].values.reshape(-1, 1))
capital_loss = capital_loss_discretizer.transform(test_samples['capital-loss'].values.reshape(-1, 1))
hours_per_week = hours_per_week_discretizer.transform(test_samples['hours-per-week'].values.reshape(-1, 1))
native_country = native_country_encoder.transform(test_samples['native-country'].values.reshape(-1, 1))
labels = test_samples.apply(lambda row: binarize(row, 'income', '>50K'), axis=1).values


with open('datasets/adult-test.csv', 'w') as file:

    file.write(f'record_id\tage\tworkclass\tfnlwgt\teducation\tmarital_status\toccupation\trelationship\trace\tsex'
               f'\tcapital_gain\thours_per_week\tnative_country\tlabel\n')

    for i in range(0, len(test_samples)):
        line = '\t'.join([
            str(i + len(train_samples)),
            str(int(age[i][0])),
            str(workclass[i]),
            str(int(fnlwgt[i][0])),
            str(education[i]),
            str(marital_status[i]),
            str(occupation[i]),
            str(relationship[i]),
            str(race[i]),
            str(sex[i]),
            str(int(capital_gain[i][0])),
            #str(int(capital_loss[i][0])),
            str(int(hours_per_week[i][0])),
            str(native_country[i]),
            str(labels[i])
        ])
        file.write(line + '\n')
