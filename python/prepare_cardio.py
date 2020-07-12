import pandas as pd

from experimentation.encoding import discretize, ordinalize, binarize

from sklearn.model_selection import train_test_split

raw_data = pd.read_csv('datasets/cardio.csv', sep=';')
raw_data = raw_data.dropna()

train_samples, test_samples = train_test_split(raw_data, test_size=0.2)

age, age_discretizer = discretize(train_samples, 'age')
gender = train_samples['gender'].values - 1
height, height_discretizer = discretize(train_samples, 'height')
weight, weight_discretizer = discretize(train_samples, 'weight')
ap_hi, ap_hi_discretizer = discretize(train_samples, 'ap_hi')
ap_lo, ap_lo_discretizer = discretize(train_samples, 'ap_lo')
cholesterol = train_samples['cholesterol'].values - 1
glucose = train_samples['gluc'].values - 1
smoke = train_samples['smoke'].values
alcohol = train_samples['alco'].values
active = train_samples['active'].values
labels = train_samples['cardio'].values

with open('datasets/cardio-train.csv', 'w') as file:

    file.write(f'record_id\tage\tgender\theight\tweight\tap_hi\tap_lo\tcholesterol\tglucose\tsmoke\talcohol\t' +
               'active\tlabel\n')

    for i in range(0, len(train_samples)):
        line = '\t'.join([
            str(i),
            str(int(age[i][0])),
            str(gender[i]),
            str(int(height[i][0])),
            str(int(weight[i][0])),
            str(int(ap_hi[i][0])),
            str(int(ap_lo[i][0])),
            str(cholesterol[i]),
            str(glucose[i]),
            str(smoke[i]),
            str(alcohol[i]),
            str(active[i]),
            str(labels[i])
        ])
        file.write(line + '\n')

age = age_discretizer.transform(test_samples['age'].values.reshape(-1, 1))
gender = test_samples['gender'].values - 1
height = height_discretizer.transform(test_samples['height'].values.reshape(-1, 1))
weight = weight_discretizer.transform(test_samples['weight'].values.reshape(-1, 1))
ap_hi = ap_hi_discretizer.transform(test_samples['ap_hi'].values.reshape(-1, 1))
ap_lo = ap_lo_discretizer.transform(test_samples['ap_lo'].values.reshape(-1, 1))
cholesterol = test_samples['cholesterol'].values - 1
glucose = test_samples['gluc'].values - 1
smoke = test_samples['smoke'].values
alcohol = test_samples['alco'].values
active = test_samples['active'].values
labels = test_samples['cardio'].values

with open('datasets/cardio-test.csv', 'w') as file:

    file.write(f'record_id\tage\tgender\theight\tweight\tap_hi\tap_lo\tcholesterol\tglucose\tsmoke\talcohol\t' +
               'active\tlabel\n')

    for i in range(0, len(test_samples)):
        line = '\t'.join([
            str(i + len(train_samples)),
            str(int(age[i][0])),
            str(gender[i]),
            str(int(height[i][0])),
            str(int(weight[i][0])),
            str(int(ap_hi[i][0])),
            str(int(ap_lo[i][0])),
            str(cholesterol[i]),
            str(glucose[i]),
            str(smoke[i]),
            str(alcohol[i]),
            str(active[i]),
            str(labels[i])
        ])
        file.write(line + '\n')
