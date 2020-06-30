import numpy as np
import pandas as pd


samples = pd.read_csv('../datasets/titanic.csv')

print(np.min(samples['Siblings/Spouses Aboard'].values), np.max(samples['Siblings/Spouses Aboard'].values))
print(np.min(samples['Parents/Children Aboard'].values), np.max(samples['Parents/Children Aboard'].values))