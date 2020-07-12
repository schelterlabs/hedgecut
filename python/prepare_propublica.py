import pandas as pd

from experimentation.encoding import discretize, ordinalize, binarize

from sklearn.model_selection import train_test_split


"""The custom pre-processing function is adapted from
https://github.com/IBM/AIF360/blob/master/aif360/algorithms/preprocessing/optim_preproc_helpers/data_preproc_functions.py
https://github.com/fair-preprocessing/nips2017/blob/master/compas/code/Generate_Compas_Data.ipynb
"""
df = pd.read_csv('datasets/propublica-recidivism.csv', na_values='?', sep=',')
df = df[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text',
         'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score',
         'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]
ix = df['days_b_screening_arrest'] <= 100
ix = (df['days_b_screening_arrest'] >= -100) & ix
ix = (df['is_recid'] != -1) & ix
ix = (df['c_charge_degree'] != "O") & ix
ix = (df['score_text'] != 'N/A') & ix
df = df.loc[ix, :]
df['length_of_stay'] = (pd.to_datetime(df['c_jail_out']) - pd.to_datetime(df['c_jail_in'])).apply(
    lambda x: x.days)

df = df[['age', 'decile_score', 'priors_count', 'days_b_screening_arrest', 'decile_score', 'is_recid',
        'c_charge_degree', 'age_cat', 'score_text', 'sex', 'race', 'two_year_recid']]

train_samples, test_samples = train_test_split(df, test_size=0.2)

age, age_discretizer = discretize(train_samples, 'age')
decile_score, decile_score_discretizer = discretize(train_samples, 'decile_score')
priors_count, priors_count_discretizer = discretize(train_samples, 'priors_count')
days_b_screening_arrest, days_b_screening_arrest_discretizer = discretize(train_samples, 'days_b_screening_arrest')
is_recid = train_samples['is_recid'].values
c_charge_degree = train_samples.apply(lambda row: binarize(row, 'c_charge_degree', 'F'), axis=1).values
sex = train_samples.apply(lambda row: binarize(row, 'sex', 'Female'), axis=1).values
age_cat, age_cat_encoder = ordinalize(df, train_samples, 'age_cat')
score_text, score_text_encoder = ordinalize(df, train_samples, 'score_text')
race, race_encoder = ordinalize(df, train_samples, 'race')
labels = train_samples['two_year_recid'].values

with open('datasets/propublica-train.csv', 'w') as file:

    file.write(f'record_id\tage\tdecile_score\tpriors_count\tdays_b_screening_arrest\tis_recid\tc_charge_degree\t' +
               'sex\tage_cat\tscore_text\trace\tlabel\n')

    for i in range(0, len(train_samples)):
        line = '\t'.join([
            str(i),
            str(int(age[i][0])),
            str(int(decile_score[i][0])),
            str(int(priors_count[i][0])),
            str(int(days_b_screening_arrest[i][0])),
            str(is_recid[i]),
            str(c_charge_degree[i]),
            str(sex[i]),
            str(age_cat[i]),
            str(score_text[i]),
            str(race[i]),
            str(labels[i])
        ])
        file.write(line + '\n')

age, age_discretizer.transform(test_samples['age'].values.reshape(-1, 1))
decile_score, decile_score_discretizer.transform(test_samples['decile_score'].values.reshape(-1, 1))
priors_count, priors_count_discretizer.transform(test_samples['priors_count'].values.reshape(-1, 1))
days_b_screening_arrest, days_b_screening_arrest_discretizer\
    .transform(test_samples['days_b_screening_arrest'].values.reshape(-1, 1))
is_recid = test_samples['is_recid'].values
c_charge_degree = test_samples.apply(lambda row: binarize(row, 'c_charge_degree', 'F'), axis=1).values
sex = test_samples.apply(lambda row: binarize(row, 'sex', 'Female'), axis=1).values
age_cat, age_cat_encoder.transform(test_samples['age_cat'].values.reshape(-1, 1))
score_text, score_text_encoder.transform(test_samples['score_text'].values.reshape(-1, 1))
race, race_encoder.transform(test_samples['race'].values.reshape(-1, 1))
labels = test_samples['two_year_recid'].values

with open('datasets/propublica-test.csv', 'w') as file:

    file.write(f'record_id\tage\tdecile_score\tpriors_count\tdays_b_screening_arrest\tis_recid\tc_charge_degree\t' +
               'sex\tage_cat\tscore_text\trace\tlabel\n')

    for i in range(0, len(test_samples)):
        line = '\t'.join([
            str(i + len(train_samples)),
            str(int(age[i][0])),
            str(int(decile_score[i][0])),
            str(int(priors_count[i][0])),
            str(int(days_b_screening_arrest[i][0])),
            str(is_recid[i]),
            str(c_charge_degree[i]),
            str(sex[i]),
            str(age_cat[i]),
            str(score_text[i]),
            str(race[i]),
            str(labels[i])
        ])
        file.write(line + '\n')
