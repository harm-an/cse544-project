#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 00:41:57 2021

@author: mayankjain
"""

import pandas as pd
import numpy as np
import math
from collections import OrderedDict


def get_outlier(raw_data, data_type, colNumber):
    data = raw_data[data_type].values.tolist()
    data.sort()
    alpha = 1.5

    q1_val = data[math.ceil(len(data) * 0.25)]

    q3_val = data[math.ceil(len(data) * 0.75)]


    inter_quartile_range = q3_val - q1_val
    low = q1_val - alpha * inter_quartile_range
    high = q3_val + alpha * inter_quartile_range
    
    print("left outlier boundary value for " + data_type ,low)
    print("right outlier boundary value for "  + data_type,high)

    #Storing all the dates for which outliers are present
    date_value=[]
    for index,row in raw_data.iterrows():
        if(row[colNumber] < low or row[colNumber] > high):
            date_value.append(row[0])

#Return the list containing outlier dates
    return date_value


raw_data = pd.read_csv('data.csv')

# Adding new columns to csv containing daily data 

raw_data['daily_GA_confirmed'] =[0]*438
raw_data['daily_HI_confirmed'] =[0]*438
raw_data['daily_GA_deaths'] =[0]*438
raw_data['daily_HI_deaths'] =[0]*438

# Calculate daily stats from the original cumulative data for each column
for index,row in raw_data.iterrows():
      raw_data['daily_GA_confirmed'] = raw_data['GA confirmed'] - raw_data['GA confirmed'].shift(1) 
      raw_data['daily_HI_confirmed'] = raw_data['HI confirmed'] - raw_data['HI confirmed'].shift(1) 
      raw_data['daily_GA_deaths'] = raw_data['GA deaths'] - raw_data['GA deaths'].shift(1) 
      raw_data['daily_HI_deaths'] = raw_data['HI deaths'] - raw_data['HI deaths'].shift(1) 

raw_data.at[0,'daily_GA_confirmed'] = 0
raw_data.at[0,'daily_HI_confirmed'] = 0
raw_data.at[0,'daily_GA_deaths'] = 0
raw_data.at[0,'daily_HI_deaths'] = 0 

# Get outlier dates of every column and add them into a list    
dates_to_be_removed = []
dates_to_be_removed.append(get_outlier(raw_data, 'daily_GA_confirmed',5))
dates_to_be_removed.append(get_outlier(raw_data, 'daily_HI_confirmed',6))
dates_to_be_removed.append(get_outlier(raw_data, 'daily_GA_deaths',7))
dates_to_be_removed.append(get_outlier(raw_data, 'daily_HI_deaths',8))

outlier_list = list(np.concatenate(dates_to_be_removed).flat)
outlier_list = list(OrderedDict.fromkeys(outlier_list))
outlier_list.sort()
print("Number of outliers detected: ", len(outlier_list) )
print("")
print("Data of these dates is detected as outlier: ", outlier_list)


# We are removing outliers only for months other than Aug 20,Oct 20,Nov 20,Dec 20,Feb 21,Mar 21

for index, row in raw_data.iterrows():
    if row['Date'] in outlier_list:
        if row['Date'][5:7] != '08' and row['Date'][5:7] != '02' and row['Date'][5:7] != '03' and row['Date'][5:7] != '10' and row['Date'][5:7] != '11' and row['Date'][5:7] != '12':
            raw_data.drop(index,inplace = True)

# Store the new dataframe in a new file that will be used in all the other parts
raw_data.to_csv("cleaned_data.csv", index=False)


