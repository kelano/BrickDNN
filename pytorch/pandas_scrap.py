import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


df = pd.read_csv('data_201903.csv')

# print(df)
# print(df.dtypes)
# print(df.head())
print(df.columns)

print()

df[(df.Theme == 'Star Wars') & (df.Minifigs > 20)]

print(df[(df.Theme == 'Star Wars') & (df.Minifigs > 20)])

# print(df['Name'], df['Theme'])

corr = df.corr(method='pearson')
plt.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()

