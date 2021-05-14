#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:31:13 2021

@author: mayankjain
"""
import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.stats import binom
from scipy.stats import geom

def OneSampleKS_Poisson(data_cases,datatype):
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
        
    
    # print("ga data: ",ga_confirmed_data )
    # print("hi data: ",hi_confirmed_data )
    
    # Lambda mme using first state data
    lambda_param = float(np.sum(ga_confirmed_data)) / float(len(ga_confirmed_data))
    
    # Find ecdf of second data
    hi_confirmed_data.sort()
    second_DataLength = len(hi_confirmed_data)
    ecdf_secondData = []
    stepSize = 1 / second_DataLength
    step = 0
    for j in range(second_DataLength - 1):
        step += stepSize
        ecdf_secondData.append(step)
    
    ecdf_secondData.append(1) 
    #print(ecdf_secondData)


    maxDifference = 0
    print("lambda_param: ",lambda_param)
    # Iterating over all the data points to calculate difference
    for k in range(second_DataLength):
        #print("hi_confirmed_data: ",hi_confirmed_data[k])
        cdfAtPoint = poisson.cdf(hi_confirmed_data[k], lambda_param)

        if k == 0:
            ecdf_Left = 0
        else:
            ecdf_Left = ecdf_secondData[k-1]

        ecdf_Right = ecdf_secondData[k]
        
        diff_ecdf_Left = abs(cdfAtPoint - ecdf_Left)
        diff_ecdf_Right = abs(cdfAtPoint - ecdf_Right)
        # print("cdfAtPoint: ", cdfAtPoint)
        # print("ecdf_Left: ", ecdf_Left)
        # print("ecdf_Right: ", ecdf_Right)
        
        difference = max(diff_ecdf_Left,diff_ecdf_Right)
        #Keep Updating maxdiff at each iteration
        if maxDifference < difference:
            maxDifference = difference

    print("\nMaximum Difference: ", maxDifference)
    c = 0.05
    # Comparing maximum difference with critical value
    if(maxDifference > c):
        print("Null hypothesis is rejected as Oct-Dec 2020 data for the second state does not have the distribution with the obtained MME parameters for ", datatype)
    else:
        print("Null hypothesis is accepted as Oct-Dec 2020 data for the second state have the distribution with the obtained MME parameters for ", datatype)
    
    
def OneSampleKS_Geometric(data_cases,datatype):
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
    
    # print("ga data: ",ga_confirmed_data )
    # print("hi data: ",hi_confirmed_data )
    
    # geometric P mme using first state data
    p_mme =  float(len(ga_confirmed_data)) / float(np.sum(ga_confirmed_data))
    
    # Find ecdf of second data
    hi_confirmed_data.sort()
    second_DataLength = len(hi_confirmed_data)
    ecdf_secondData = []
    stepSize = 1 / second_DataLength
    step = 0
    for j in range(second_DataLength - 1):
        step += stepSize
        ecdf_secondData.append(step)
    
    ecdf_secondData.append(1) 
    #print(ecdf_secondData)


    maxDifference = 0
    print("p_mme: ",p_mme)
    # Iterating over all the data points to calculate difference
    for k in range(second_DataLength):
        #print("hi_confirmed_data: ",hi_confirmed_data[k])
        cdfAtPoint = geom.cdf(hi_confirmed_data[k], p_mme)

        if k == 0:
            ecdf_Left = 0
        else:
            ecdf_Left = ecdf_secondData[k-1]

        ecdf_Right = ecdf_secondData[k]
        
        diff_ecdf_Left = abs(cdfAtPoint - ecdf_Left)
        diff_ecdf_Right = abs(cdfAtPoint - ecdf_Right)
        # print("cdfAtPoint: ", cdfAtPoint)
        # print("ecdf_Left: ", ecdf_Left)
        # print("ecdf_Right: ", ecdf_Right)
        
        difference = max(diff_ecdf_Left,diff_ecdf_Right)
        if maxDifference < difference:
            maxDifference = difference

    print("\nMaximum Difference: ", maxDifference)
    c = 0.05
    # Comparing maximum difference with critical value
    if(maxDifference > c):
        print("Null hypothesis is rejected as Oct-Dec 2020 data for the second state does not have the distribution with the obtained MME parameters for ", datatype)
    else:
        print("Null hypothesis is accepted as Oct-Dec 2020 data for the second state have the distribution with the obtained MME parameters for ", datatype)
        
        
def OneSampleKS_Binomial(data_cases,datatype):
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
    
    # print("ga data: ",ga_confirmed_data )
    # print("hi data: ",hi_confirmed_data )
    
    # Binomial p_mme and n_mme using first state data
    sample_variance = np.var(ga_confirmed_data)
    sample_mean = float(np.sum(ga_confirmed_data)) / float(len(ga_confirmed_data))
    n_mme = pow(sample_mean,2) / (sample_mean - sample_variance)
    p_mme =  1 - (sample_variance/sample_mean)
    
    # Find ecdf of second data
    hi_confirmed_data.sort()
    second_DataLength = len(hi_confirmed_data)
    ecdf_secondData = []
    stepSize = 1 / second_DataLength
    step = 0
    for j in range(second_DataLength - 1):
        step += stepSize
        ecdf_secondData.append(step)
    
    ecdf_secondData.append(1) 
    #print(ecdf_secondData)


    maxDifference = 0
    print("p_mme: ",p_mme)
    print("n_mme: ",n_mme)
    # Iterating over all the data points to calculate difference
    for k in range(second_DataLength):
        # print("hi_confirmed_data: ",hi_confirmed_data[k])
        cdfAtPoint = binom.cdf(hi_confirmed_data[k], n_mme, p_mme)

        if k == 0:
            ecdf_Left = 0
        else:
            ecdf_Left = ecdf_secondData[k-1]

        ecdf_Right = ecdf_secondData[k]
        
        diff_ecdf_Left = abs(cdfAtPoint - ecdf_Left)
        diff_ecdf_Right = abs(cdfAtPoint - ecdf_Right)
        # print("cdfAtPoint: ", cdfAtPoint)
        # print("ecdf_Left: ", ecdf_Left)
        # print("ecdf_Right: ", ecdf_Right)
        
        difference = max(diff_ecdf_Left,diff_ecdf_Right)
        if maxDifference < difference:
            maxDifference = difference

    print("\nMaximum Difference: ", maxDifference)
    c = 0.05
    # Comparing maximum difference with critical value
    if(maxDifference > c):
        print("Null hypothesis is rejected as Oct-Dec 2020 data for the second state does not have the distribution with the obtained MME parameters for ", datatype)
    else:
        print("Null hypothesis is accepted as Oct-Dec 2020 data for the second state has the distribution with the obtained MME parameters for ", datatype)
    



var1 = ['Date','daily_GA_confirmed','daily_HI_confirmed']   
data_cases = pd.read_csv('cleaned_data.csv',usecols = var1)     
datatype = 'Confirmed cases'

print("1 Sample K-S test: Checking equality of distributions for confirmed cases in 2 states using Poisson distribution")
OneSampleKS_Poisson(data_cases,datatype)
print("")
print("1 Sample K-S test: Checking equality of distributions for confirmed cases in 2 states using Geometric distribution")
OneSampleKS_Geometric(data_cases,datatype)
print("")
print("1 Sample K-S test: Checking equality of distributions for confirmed cases in 2 states using Binomial distribution")
OneSampleKS_Binomial(data_cases,datatype)



var2 = ['Date','daily_GA_deaths','daily_HI_deaths']   
data_deaths = pd.read_csv('cleaned_data.csv',usecols = var2)     
datatype = 'Deaths'
print("")
print("1 Sample K-S test: Checking equality of distributions for deaths in 2 states using Poisson distribution")
OneSampleKS_Poisson(data_deaths,datatype)
print("")
print("1 Sample K-S test: Checking equality of distributions for deaths in 2 states using Geometric distribution")
OneSampleKS_Geometric(data_deaths,datatype)
print("")
print("1 Sample K-S test: Checking equality of distributions for deaths in 2 states using Binomial distribution")
OneSampleKS_Binomial(data_deaths,datatype)

