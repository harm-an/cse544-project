# Hyopthesis 2: Industry related air pollution in Texas is related to COVID 19
import pandas as pd
import numpy as np
from pearson_test_module import pearson_test

so2 = pd.read_csv('pollution_texas_SO2.csv')
covidCases = pd.read_csv('US_confirmed.csv' , index_col=0)

covidCases = covidCases.transpose()
covidCases_data = []

for index, row in covidCases.iterrows():
    covidCases_data.append(row['TX'])

covidCases_data = covidCases_data[60:359]
so2_data = []
for index, row in so2.iterrows():
    if (row["Site Name"] == 'Midlothian OFW'):
        so2_data.append(row["Daily Max 1-hour SO2 Concentration"])

so2_data = so2_data[60:]
pearson_coeff = pearson_test(covidCases_data, so2_data)

print("\nPearson Coefficient for Covid Cases in 2020 vs SO2 emission in 2020 in TX):\n", pearson_coeff)

