# Hyopthesis 1: Traffic related air pollution in Texas is related to COVID 19
import pandas as pd
import numpy as np
from pearson_test_module import pearson_test

no2 = pd.read_csv('pollution_texas_NO2.csv')
covidCases = pd.read_csv('US_confirmed.csv' , index_col=0)

covidCases = covidCases.transpose()
covidCases_data = []

for index, row in covidCases.iterrows():
    covidCases_data.append(row['TX'])

covidCases_data = covidCases_data[81:366]

no2Data = []
for index, row in no2.iterrows():
    if (row["Site Name"] == 'Arlington Municipal Airport'):
        no2Data.append(row["Daily Max 1-hour NO2 Concentration"])

pearson_coeff = pearson_test(covidCases_data, no2Data)

print("\nPearson Coefficient for Covid Cases in 2020 vs NO2 emission in 2020 in TX):\n", pearson_coeff)

