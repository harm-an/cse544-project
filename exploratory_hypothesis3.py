# Hyopthesis 3: Industry related air pollution in Texas is related to COVID 19
import pandas as pd
import numpy as np
from pearson_test_module import pearson_test

co = pd.read_csv('pollution_texas_CO.csv')
covidCases = pd.read_csv('US_confirmed.csv' , index_col=0)

covidCases = covidCases.transpose()
covidCases_data = []

for index, row in covidCases.iterrows():
    covidCases_data.append(row['TX'])

covidCases_data = covidCases_data[0:362]

co_data = []

for index, row in co.iterrows():
    if (row["Site Name"] == 'San Antonio Interstate 35'):
        co_data.append(row["Daily Max 8-hour CO Concentration"])

pearson_coeff = pearson_test(covidCases_data, co_data)


print("\nPearson Coefficient for Covid Cases in 2020 vs CO concentration in 2020 in TX):\n", pearson_coeff)

