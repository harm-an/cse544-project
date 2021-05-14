#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 01:43:49 2021

@author: mayankjain
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import copy

def calculate_ecdf(x):
    n = len(x)
    x.sort()
    y = []
    yi = 0
    
    for i in x:
        yi += 1/n
        y.append(yi)
    y[-1] = 1
    return (x, y)

def find_diff(data_cases,datatype):
    #Reusing 2 sample KS test from our previous assignment submission
    row,col = data_cases.shape

    ga_confirmed_data = []
    hi_confirmed_data = []
    
    for index, row in data_cases.iterrows():
        # Oct,Nov,Dec data for both states
        if row['Date'][5] =='1':
            if datatype == 'Confirmed cases':
                ga_confirmed_data.append(row['daily_GA_confirmed'])
                hi_confirmed_data.append(row['daily_HI_confirmed'])
            
            else:
                ga_confirmed_data.append(row['daily_GA_deaths'])
                hi_confirmed_data.append(row['daily_HI_deaths'])


    ga_cases, ga_cdf = calculate_ecdf(ga_confirmed_data)
    hi_cases, hi_cdf = calculate_ecdf(hi_confirmed_data)
    
    #print("ga_cdf: ",ga_cdf)
    #print("hi_cdf: ",hi_cdf)

    cols = dict()
    cols["ga_minus"] = 0;
    cols["ga_plus"] = 0;
    cols["hi_minus"] = 0;
    cols["hi_plus"] = 0;

    table = []

    j = 0

    mx_diff = 0
    index = 0
    for i in range(len(ga_confirmed_data)):
        col_temp = copy.copy(cols)
        if (i == 0):
            col_temp["ga_minus"] = 0
        else:
            col_temp["ga_minus"] = table[i-1]["ga_plus"]
 
        col_temp["ga_plus"] = ga_cdf[i]

        while(j<len(hi_cases) and hi_cases[j] < ga_cases[i]):
            j+=1
        
        if j != len(hi_cases):
            col_temp["hi_minus"] = hi_cdf[j-1]

            if (hi_cases[j] == ga_cases[i]):
                col_temp["hi_plus"] = hi_cdf[j];
            else:
                col_temp["hi_plus"] = hi_cdf[j-1];
        else:
            col_temp["hi_minus"] = 0
            col_temp["hi_plus"] = 0

        diff1 = abs(col_temp["ga_minus"] - col_temp["hi_minus"])
        diff2 = abs(col_temp["ga_plus"] - col_temp["hi_plus"])

        diff = max(diff1, diff2)
        if (mx_diff < diff):
            mx_diff = diff
            index = i

        table.append(col_temp)
        
        
    print("Max Diff is: ", mx_diff)
    # Comparing maximum difference with critical value
    if(mx_diff > 0.05):
        print("Null hypothesis that 2 states have same distribution for ", datatype," is rejected.")
    else:
        print("Null hypothesis that 2 states have same distribution for ", datatype," is accepted.")



var1 = ['Date','daily_GA_confirmed','daily_HI_confirmed']   
data = pd.read_csv('cleaned_data.csv',usecols = var1) 
datatype = 'Confirmed cases'
print("Checking equality of distributions for confirmed cases in 2 states using two sample KS test.")
find_diff(data,datatype)


var2 = ['Date','daily_GA_deaths','daily_HI_deaths']   
data = pd.read_csv('cleaned_data.csv',usecols = var2) 
datatype = 'Deaths'
print("")
print("Checking equality of distributions for deaths in 2 states using two sample KS test.")
find_diff(data,datatype)

