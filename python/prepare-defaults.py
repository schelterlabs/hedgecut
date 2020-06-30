import numpy as np
import pandas as pd
import duckdb

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

def discretize(data, attribute):
    discretizer = KBinsDiscretizer(n_bins=16, encode='ordinal', strategy='quantile')
    discretizer = discretizer.fit(data[attribute].values.reshape(-1, 1))
    transformed_values = discretizer.transform(data[attribute].values.reshape(-1, 1))
    return transformed_values, discretizer

conn = duckdb.connect()

conn.execute("CREATE TABLE defaults_raw AS SELECT * FROM read_csv_auto('../datasets/defaults.csv');")

raw_data = conn.execute("SELECT * FROM defaults_raw").fetchdf()

train_samples, test_samples = train_test_split(raw_data, test_size=0.2)



# ID,LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6,default payment next month

limit, limit_discretizer = discretize(train_samples, 'LIMIT_BAL')
sex = train_samples['SEX'].values
education = train_samples['EDUCATION'].values
marriage = train_samples['MARRIAGE'].values
age, age_discretizer = discretize(train_samples, 'AGE')
pay0 = train_samples['PAY_0'].values + 2
pay2 = train_samples['PAY_2'].values + 2
pay3 = train_samples['PAY_3'].values + 2
pay4 = train_samples['PAY_4'].values + 2
pay5 = train_samples['PAY_5'].values + 2
pay6 = train_samples['PAY_6'].values + 2
bill_amt1, bill_amt1_discretizer = discretize(train_samples, 'BILL_AMT1')
bill_amt2, bill_amt2_discretizer = discretize(train_samples, 'BILL_AMT2')
bill_amt3, bill_amt3_discretizer = discretize(train_samples, 'BILL_AMT3')
bill_amt4, bill_amt4_discretizer = discretize(train_samples, 'BILL_AMT4')
bill_amt5, bill_amt5_discretizer = discretize(train_samples, 'BILL_AMT5')
bill_amt6, bill_amt6_discretizer = discretize(train_samples, 'BILL_AMT6')
pay_amt1, pay_amt1_discretizer = discretize(train_samples, 'PAY_AMT1')
pay_amt2, pay_amt2_discretizer = discretize(train_samples, 'PAY_AMT2')
pay_amt3, pay_amt3_discretizer = discretize(train_samples, 'PAY_AMT3')
pay_amt4, pay_amt4_discretizer = discretize(train_samples, 'PAY_AMT4')
pay_amt5, pay_amt5_discretizer = discretize(train_samples, 'PAY_AMT5')
pay_amt6, pay_amt6_discretizer = discretize(train_samples, 'PAY_AMT6')
label = train_samples['default payment next month'].values

with open('../datasets/defaults-train.csv', 'w') as file:

    file.write(f'record_id\tlimit\tsex\teducation\tmarriage\tage\tpay0\tpay2\tpay3\tpay4\tpay5\tpay6\tbill_amt1\tbill_amt2\tbill_amt3\tbill_amt4\tbill_amt5\tbill_amt6\tpay_amt1\tpay_amt2\tpay_amt3\tpay_amt4\tpay_amt5\tpay_amt6\tlabel\n')

    for i in range(0, len(train_samples)):
        line = '\t'.join([
            str(i),
            str(int(limit[i][0])),
            str(sex[i]),
            str(education[i]),
            str(marriage[i]),
            str(int(age[i][0])),
            str(pay0[i]),
            str(pay2[i]),
            str(pay3[i]),
            str(pay4[i]),
            str(pay5[i]),
            str(pay6[i]),
            str(int(bill_amt1[i][0])),
            str(int(bill_amt2[i][0])),
            str(int(bill_amt3[i][0])),
            str(int(bill_amt4[i][0])),
            str(int(bill_amt5[i][0])),
            str(int(bill_amt6[i][0])),
            str(int(pay_amt1[i][0])),
            str(int(pay_amt2[i][0])),
            str(int(pay_amt3[i][0])),
            str(int(pay_amt4[i][0])),
            str(int(pay_amt5[i][0])),
            str(int(pay_amt6[i][0])),
            str(label[i])
        ])
        file.write(line + '\n')
#print(raw_data)

limit = limit_discretizer.transform(test_samples['LIMIT_BAL'].values.reshape(-1, 1))
sex = test_samples['SEX'].values
education = test_samples['EDUCATION'].values
marriage = test_samples['MARRIAGE'].values
age = age_discretizer.transform(test_samples['AGE'].values.reshape(-1, 1))
pay0 = test_samples['PAY_0'].values + 2
pay2 = test_samples['PAY_2'].values + 2
pay3 = test_samples['PAY_3'].values + 2
pay4 = test_samples['PAY_4'].values + 2
pay5 = test_samples['PAY_5'].values + 2
pay6 = test_samples['PAY_6'].values + 2
bill_amt1 = bill_amt1_discretizer.transform(test_samples['BILL_AMT1'].values.reshape(-1, 1))
bill_amt2 = bill_amt2_discretizer.transform(test_samples['BILL_AMT2'].values.reshape(-1, 1))
bill_amt3 = bill_amt3_discretizer.transform(test_samples['BILL_AMT3'].values.reshape(-1, 1))
bill_amt4 = bill_amt4_discretizer.transform(test_samples['BILL_AMT4'].values.reshape(-1, 1))
bill_amt5 = bill_amt5_discretizer.transform(test_samples['BILL_AMT5'].values.reshape(-1, 1))
bill_amt6 = bill_amt6_discretizer.transform(test_samples['BILL_AMT6'].values.reshape(-1, 1))
pay_amt1 = pay_amt1_discretizer.transform(test_samples['PAY_AMT1'].values.reshape(-1, 1))
pay_amt2 = pay_amt2_discretizer.transform(test_samples['PAY_AMT2'].values.reshape(-1, 1))
pay_amt3 = pay_amt3_discretizer.transform(test_samples['PAY_AMT3'].values.reshape(-1, 1))
pay_amt4 = pay_amt4_discretizer.transform(test_samples['PAY_AMT4'].values.reshape(-1, 1))
pay_amt5 = pay_amt5_discretizer.transform(test_samples['PAY_AMT5'].values.reshape(-1, 1))
pay_amt6 = pay_amt6_discretizer.transform(test_samples['PAY_AMT6'].values.reshape(-1, 1))
label = test_samples['default payment next month'].values

with open('../datasets/defaults-test.csv', 'w') as file:

    file.write(f'record_id\tlimit\tsex\teducation\tmarriage\tage\tpay0\tpay2\tpay3\tpay4\tpay5\tpay6\tbill_amt1\tbill_amt2\tbill_amt3\tbill_amt4\tbill_amt5\tbill_amt6\tpay_amt1\tpay_amt2\tpay_amt3\tpay_amt4\tpay_amt5\tpay_amt6\tlabel\n')

    for i in range(0, len(test_samples)):
        line = '\t'.join([
            str(i + len(train_samples)),
            str(int(limit[i][0])),
            str(sex[i]),
            str(education[i]),
            str(marriage[i]),
            str(int(age[i][0])),
            str(pay0[i]),
            str(pay2[i]),
            str(pay3[i]),
            str(pay4[i]),
            str(pay5[i]),
            str(pay6[i]),
            str(int(bill_amt1[i][0])),
            str(int(bill_amt2[i][0])),
            str(int(bill_amt3[i][0])),
            str(int(bill_amt4[i][0])),
            str(int(bill_amt5[i][0])),
            str(int(bill_amt6[i][0])),
            str(int(pay_amt1[i][0])),
            str(int(pay_amt2[i][0])),
            str(int(pay_amt3[i][0])),
            str(int(pay_amt4[i][0])),
            str(int(pay_amt5[i][0])),
            str(int(pay_amt6[i][0])),
            str(label[i])
        ])
        file.write(line + '\n')