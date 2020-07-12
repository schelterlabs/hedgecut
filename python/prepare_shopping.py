import numpy as np
import pandas as pd

from experimentation.encoding import discretize, ordinalize, binarize

from sklearn.model_selection import train_test_split


raw_data = pd.read_csv('datasets/shopping.csv', sep=',', index_col=False)
raw_data = raw_data.dropna()
raw_data['Weekend'] = raw_data['Weekend'].astype(str)
raw_data['Revenue'] = raw_data['Revenue'].astype(str)

train_samples, test_samples = train_test_split(raw_data, test_size=0.2)

administrative, administrative_discretizer = discretize(train_samples, 'Administrative')
administrative_duration, administrative_duration_discretizer = discretize(train_samples, 'Administrative_Duration')
informational, informational_discretizer = discretize(train_samples, 'Informational')
informational_duration, informational_duration_discretizer = discretize(train_samples, 'Informational_Duration')
product_related, product_related_discretizer = discretize(train_samples, 'ProductRelated')
product_related_duration, product_related_duration_discretizer = discretize(train_samples, 'ProductRelated_Duration')
bounce_rates, bounce_rates_discretizer = discretize(train_samples, 'BounceRates')
exit_rates, exit_rates_discretizer = discretize(train_samples, 'ExitRates')
page_values, page_values_discretizer = discretize(train_samples, 'PageValues')
special_day, special_day_discretizer = discretize(train_samples, 'SpecialDay')
month, month_encoder = ordinalize(raw_data, train_samples, 'Month')
operating_systems = train_samples['OperatingSystems'].values - 1
browser = train_samples['Browser'].values - 1
region = train_samples['Region'].values - 1
traffic_type = train_samples['TrafficType'].values - 1
visitor_type, visitor_type_encoder = ordinalize(raw_data, train_samples, 'VisitorType')
weekend = train_samples.apply(lambda row: binarize(row, 'Weekend', 'True'), axis=1).values
labels = train_samples.apply(lambda row: binarize(row, 'Revenue', 'True'), axis=1).values

with open('datasets/shopping-train.csv', 'w') as file:

    file.write(f'record_id\tadministrative\tadministrative_duration\tinformational\tinformational_duration' +
               f'\tproduct_related\tproduct_related_duration\tbounce_rates\texit_rates\tpage_values\t' +
               f'special_day\tmonth\toperating_systems\tbrowser\tregion\ttraffic_type\tvisitor_type\t' +
               f'weekend\tlabel\n')

    for i in range(0, len(train_samples)):
        line = '\t'.join([
            str(i),
            str(int(administrative[i][0])),
            str(int(administrative_duration[i][0])),
            str(int(informational[i][0])),
            str(int(informational_duration[i][0])),
            str(int(product_related[i][0])),
            str(int(product_related_duration[i][0])),
            str(int(bounce_rates[i][0])),
            str(int(exit_rates[i][0])),
            str(int(page_values[i][0])),
            str(int(special_day[i][0])),
            str(month[i]),
            str(operating_systems[i]),
            str(browser[i]),
            str(region[i]),
            str(traffic_type[i]),
            str(visitor_type[i]),
            str(weekend[i]),
            str(labels[i])
        ])
        file.write(line + '\n')

administrative = administrative_discretizer.transform(test_samples['Administrative'].values.reshape(-1, 1))
administrative_duration = administrative_duration_discretizer.transform(test_samples['Administrative_Duration']
                                                                        .values.reshape(-1, 1))
informational = informational_discretizer.transform(test_samples['Informational'].values.reshape(-1, 1))
informational_duration = informational_duration_discretizer.transform(test_samples['Informational_Duration']
                                                                      .values.reshape(-1, 1))
product_related = product_related_discretizer.transform(test_samples['ProductRelated'].values.reshape(-1, 1))
product_related_duration = product_related_duration_discretizer.transform(test_samples['ProductRelated_Duration']
                                                                          .values.reshape(-1, 1))
bounce_rates = bounce_rates_discretizer.transform(test_samples['BounceRates'].values.reshape(-1, 1))
exit_rates = exit_rates_discretizer.transform(test_samples['ExitRates'].values.reshape(-1, 1))
page_values = page_values_discretizer.transform(test_samples['PageValues'].values.reshape(-1, 1))
special_day = special_day_discretizer.transform(test_samples['SpecialDay'].values.reshape(-1, 1))
month = month_encoder.transform(test_samples['Month'].values.reshape(-1, 1))
operating_systems = train_samples['OperatingSystems'].values - 1
browser = test_samples['Browser'].values - 1
region = test_samples['Region'].values - 1
traffic_type = test_samples['TrafficType'].values - 1
visitor_type = visitor_type_encoder.transform(test_samples['VisitorType'].values.reshape(-1, 1))
weekend = test_samples.apply(lambda row: binarize(row, 'Weekend', 'True'), axis=1).values
labels = test_samples.apply(lambda row: binarize(row, 'Revenue', 'True'), axis=1).values

with open('datasets/shopping-test.csv', 'w') as file:

    file.write(f'record_id\tadministrative\tadministrative_duration\tinformational\tinformational_duration' +
               f'\tproduct_related\tproduct_related_duration\tbounce_rates\texit_rates\tpage_values\t' +
               f'special_day\tmonth\toperating_systems\tbrowser\tregion\ttraffic_type\tvisitor_type\t' +
               f'weekend\tlabel\n')

    for i in range(0, len(test_samples)):
        line = '\t'.join([
            str(i + len(train_samples)),
            str(int(administrative[i][0])),
            str(int(administrative_duration[i][0])),
            str(int(informational[i][0])),
            str(int(informational_duration[i][0])),
            str(int(product_related[i][0])),
            str(int(product_related_duration[i][0])),
            str(int(bounce_rates[i][0])),
            str(int(exit_rates[i][0])),
            str(int(page_values[i][0])),
            str(int(special_day[i][0])),
            str(month[i]),
            str(operating_systems[i]),
            str(browser[i]),
            str(region[i]),
            str(traffic_type[i]),
            str(visitor_type[i]),
            str(weekend[i]),
            str(labels[i])
        ])
        file.write(line + '\n')