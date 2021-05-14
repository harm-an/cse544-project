#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 12:17:52 2021

@author: mayankjain
"""

import pandas as pd
import numpy as np
import random


def getp_val(confirmed_data,len_confirmed_data,observed_T,num_permutation):

#Reusing permutation test code from our previous assignment submission
    count = 0
    trials = 0
    
    while trials <  num_permutation:
        temp_data = np.array(confirmed_data)
        random.shuffle((temp_data))
        
        d1 = temp_data[0:len_confirmed_data]
        d2 = temp_data[len_confirmed_data:]
        
        
        mean_d1 = np.mean(d1)
        mean_d2 = np.mean(d2)
        
        mean_diff = abs(mean_d1 - mean_d2)
        
        #Calculate count of values greater than observed_T
        if mean_diff > observed_T:
            count+=1

        trials+=1
    
    p_val = count/num_permutation
    return p_val
 

#Run permutation test for daily confirmed data
var1 = ['Date','daily_GA_confirmed','daily_HI_confirmed']   
data_cases = pd.read_csv('cleaned_data.csv',usecols = var1)

row,col = data_cases.shape

ga_confirmed_data = []
hi_confirmed_data = []

for index, row in data_cases.iterrows():
    # Fetching OCt, Nov, Dec data
    if row['Date'][5] =='1':
        ga_confirmed_data.append(row['daily_GA_confirmed'])
        hi_confirmed_data.append(row['daily_HI_confirmed'])
   

len_ga_confirmed_data = len(ga_confirmed_data)
len_hi_confirmed_data = len(hi_confirmed_data)



mean_of_ga_confirmed_data = np.mean(ga_confirmed_data)

mean_of_hi_confirmed_data = np.mean(hi_confirmed_data)



observed_T = abs(mean_of_ga_confirmed_data - mean_of_hi_confirmed_data)

print("Permutation test for daily confirmed data for Georgia and Hawaii")
print("observed_T= ",observed_T)
print("alpha = ",0.05)

confirmed_data = np.concatenate((ga_confirmed_data, hi_confirmed_data))

p_val = getp_val(confirmed_data,len_ga_confirmed_data,observed_T,1000)
print("For n = 1000 random permutations, p_value: ", p_val)
print("Therefore, NULL hypothesis for ",1000, "permutations can be rejected as p-value is less than alpha")

print("")


print("")
#Run permutation test for daily deaths data
print("Permutation test for daily deaths data for Georgia and Hawaii")

var2 = ['Date','daily_GA_deaths','daily_HI_deaths']
data_deaths = pd.read_csv('cleaned_data.csv',usecols = var2)

row,col = data_deaths.shape

ga_deaths_data = []
hi_deaths_data = []

for index, row in data_deaths.iterrows():
    # Fetching OCt, Nov, Dec data
    if row['Date'][5] =='1':
        ga_deaths_data.append(row['daily_GA_deaths'])
        hi_deaths_data.append(row['daily_HI_deaths'])


len_ga_deaths_data = len(ga_deaths_data)
len_hi_deaths_data = len(hi_deaths_data)



mean_of_ga_deaths_data = np.mean(ga_deaths_data)
mean_of_hi_deaths_data = np.mean(hi_deaths_data)



observed_T = abs(mean_of_ga_deaths_data - mean_of_hi_deaths_data)

print("observed_T= ",observed_T)
print("alpha = ",0.05)

confirmed_data = np.concatenate((ga_deaths_data, hi_deaths_data))

p_val = getp_val(confirmed_data,len_ga_deaths_data,observed_T,1000)
print("For n = 1000 random permutations, p_value: ", p_val)
print("Therefore, NULL hypothesis for ",1000, "permutations can be rejected as p-value is less than alpha")


 