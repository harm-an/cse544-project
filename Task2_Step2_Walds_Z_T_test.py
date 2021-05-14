#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:12:48 2021

@author: mayankjain
"""


import pandas as pd
import numpy as np

#1 sample wald's test
def waldsTest_1sample(feb21_data,mar21_data):
    
    theta0 = np.mean(feb21_data)  #sample mean using feb data which will work as a guess for mar data
    
    theta_cap = np.mean(mar21_data)  # As the data is Poisson distributed, so theta_cap MLE is same as sample mean
    
    data_length = float(len(mar21_data))
    standard_error = pow( theta_cap / data_length,0.5)  # As the data is Poisson distributed, so variance is theta_cap MLE
    
    wald_statistic = (theta_cap - theta0)/standard_error
    
    print("Absolute wald's statistic value: ", abs(wald_statistic))
    
    zAlphaBy2 = 1.96 #alpha = 0.05
    
    if abs(wald_statistic) > zAlphaBy2:
        print("Since wald_statistic is greater than critical value, we reject the NULL hypothesis that mean of Feb 21 data is same as Mar 21 data")
    else:
        print("Since wald_statistic is samller/equal than critical value, we accept the NULL hypothesis that mean of Feb 21 data is same as Mar 21 data")
    
    
#1 sample Z test
def Ztest_1sample(feb21_data,mar21_data,totalData):
    
    guess_mean = np.mean(feb21_data)  # mean using feb data which will work as a guess for mar data
    
    sample_mean = np.mean(mar21_data)  # Sample mean for March data
    
    #We have to calculate corrected standard deviation of whole data as mentioned in question
    totalMean = np.mean(totalData)
    totalDataLength = len(totalData)
    
    corrected_var = 0
    for row in totalData:
        corrected_var += ((row - totalMean) ** 2)/((totalDataLength - 1 )*1.0)

    marLength = len(mar21_data)
    denominator = pow(corrected_var,0.5) / pow(marLength,0.5)
    Z_statistic = (sample_mean - guess_mean)/denominator
    
    print("Absolute Z- statistic value: ", abs(Z_statistic))
    
    zAlphaBy2 = 1.96 #alpha = 0.05
    
    if abs(Z_statistic) > zAlphaBy2:
        print("Since Z- statistic is greater than critical value, we reject the NULL hypothesis that mean of Feb 21 data is same as Mar 21 data")
    else:
        print("Since Z- statistic is samller/equal than critical value, we accept the NULL hypothesis that mean of Feb 21 data is same as Mar 21 data")
   

#1 sample T test
def Ttest_1sample(feb21_data,mar21_data):
    
    guess_mean = np.mean(feb21_data)  # mean using feb data which will work as a guess for mar data
    
    sample_mean = np.mean(mar21_data)  # Sample mean for March data
    
    marLength = len(mar21_data)
    # Apply T test formula from lectures
    corrected_var = 0
    for row in mar21_data:
        corrected_var += ((row - sample_mean) ** 2)/((marLength - 1 )*1.0)


    denominator = pow(corrected_var,0.5) / pow(marLength,0.5)
    T_statistic = (sample_mean - guess_mean)/denominator
    
    print("Absolute T- statistic value: ", abs(T_statistic))
    
    critical_value = 2.3596 # t value for n - 1 = 30, alpha = 0.05
    
    if abs(T_statistic) > critical_value:
        print("Since T- statistic is greater than critical value, we reject the NULL hypothesis that mean of Feb 21 data is same as Mar 21 data")
    else:
        print("Since T- statistic is samller/equal than critical value, we accept the NULL hypothesis that mean of Feb 21 data is same as Mar 21 data")
    


# Two sample wald's test
def waldsTest_2sample(feb21_data,mar21_data):
    
    data1_mean = np.mean(feb21_data)  #sample mean using feb data 
    
    data2_mean = np.mean(mar21_data)  # sample mean using mar data 
    
    feb_length = float(len(feb21_data))
    mar_length = float(len(mar21_data))
    
    standard_error = pow((data1_mean/feb_length) + (data2_mean/mar_length), 0.5)  # As the data is Poisson distributed, so variance is sample_mean MLE
    
    wald_statistic = (data1_mean - data2_mean)/standard_error
    
    print("Absolute 2 sample wald's statistic value: ", abs(wald_statistic))
    
    critical_value = 1.96
    
    if abs(wald_statistic) > critical_value:
        print("Since wald_statistic is greater than critical value, we reject the NULL hypothesis that mean of Feb 21 data is same as Mar 21 data")
    else:
        print("Since wald_statistic is samller/equal than critical value, we accept the NULL hypothesis that mean of Feb 21 data is same as Mar 21 data")



#2 sample unpaired T test
def Ttest_2sample(feb21_data,mar21_data):
    
    data1_mean = np.mean(feb21_data)  #sample mean using feb data 
    
    data2_mean = np.mean(mar21_data)  # sample mean using mar data 
    
    feb_length = float(len(feb21_data))
    mar_length = float(len(mar21_data))
    
    corrected_var_data1 = 0
    for row in feb21_data:
        corrected_var_data1 += ((row - data1_mean) ** 2)/((feb_length - 1 )*1.0)
    
    corrected_var_data2 = 0
    for row in mar21_data:
        corrected_var_data2 += ((row - data2_mean) ** 2)/((mar_length - 1 )*1.0)


    denominator = pow((corrected_var_data1/feb_length) + (corrected_var_data2/mar_length),0.5)
    
    T_statistic = (data1_mean - data2_mean)/denominator
    
    print("Absolute 2 sample unpaired T- statistic value: ", abs(T_statistic))
    
    critical_value = 2.3022 #t value for m+n-2 = 57, alpha = 0.05
    
    if abs(T_statistic) > critical_value:
        print("Since T- statistic is greater than critical value, we reject the NULL hypothesis that mean of Feb 21 data is same as Mar 21 data")
    else:
        print("Since T- statistic is samller/equal than critical value, we accept the NULL hypothesis that mean of Feb 21 data is same as Mar 21 data")




data = pd.read_csv('cleaned_data.csv')

feb21_GA_cases = []
mar21_GA_cases = []
feb21_HI_cases = []
mar21_HI_cases = []
feb21_GA_deaths = []
mar21_GA_deaths = []
feb21_HI_deaths = []
mar21_HI_deaths = []

total_data_GA_cases = []
total_data_HI_cases = []
total_data_GA_deaths = []
total_data_HI_deaths = []

# Fetching february and march daily data for each state
for index,row in data.iterrows():
    total_data_GA_cases.append(row['daily_GA_confirmed'])
    total_data_HI_cases.append(row['daily_HI_confirmed'])
    total_data_GA_deaths.append(row['daily_GA_deaths'])
    total_data_HI_deaths.append(row['daily_HI_deaths'])
    if row['Date'][6] == '2':
        feb21_GA_cases.append(row['daily_GA_confirmed'])
        feb21_HI_cases.append(row['daily_HI_confirmed'])
        feb21_GA_deaths.append(row['daily_GA_deaths'])
        feb21_HI_deaths.append(row['daily_HI_deaths'])
    elif row['Date'][6] == '3':
        mar21_GA_cases.append(row['daily_GA_confirmed'])
        mar21_HI_cases.append(row['daily_HI_confirmed'])
        mar21_GA_deaths.append(row['daily_GA_deaths'])
        mar21_HI_deaths.append(row['daily_HI_deaths'])
            

#Check null hypothesis of GA confirmed cases for feb and mar 21 using WALD'S TEST  
print("Wald's 1 sample test for GA confirmed cases")         
waldsTest_1sample(feb21_GA_cases,mar21_GA_cases)


#Check null hypothesis of HI confirmed cases for feb and mar 21  using WALD'S TEST  
print("")
print("Wald's 1 sample test for HI confirmed cases")         
waldsTest_1sample(feb21_HI_cases,mar21_HI_cases)

#Check null hypothesis of GA deaths for feb and mar 21 using WALD'S TEST  
print("")   
print("Wald's 1 sample test for GA deaths")         
waldsTest_1sample(feb21_GA_deaths,mar21_GA_deaths)


#Check null hypothesis of HI deaths for feb and mar 21 using WALD'S TEST  
print("")  
print("Wald's 1 sample test for HI deaths")         
waldsTest_1sample(feb21_HI_deaths,mar21_HI_deaths)


#Check null hypothesis of GA confirmed cases for feb and mar 21 using Z-TEST  
print("")
print("1 sample Z-test for GA confirmed cases")         
Ztest_1sample(feb21_GA_cases,mar21_GA_cases,total_data_GA_cases)


#Check null hypothesis of HI confirmed cases for feb and mar 21  using Z-TEST   
print("")
print("1 sample Z-test for HI confirmed cases")         
Ztest_1sample(feb21_HI_cases,mar21_HI_cases,total_data_HI_cases)

#Check null hypothesis of GA deaths for feb and mar 21 using Z-TEST  
print("")   
print("1 sample Z-test for GA deaths")         
Ztest_1sample(feb21_GA_deaths,mar21_GA_deaths,total_data_GA_deaths)


#Check null hypothesis of HI deaths for feb and mar 21 using Z-TEST  
print("")  
print("1 sample Z-test for HI deaths")         
Ztest_1sample(feb21_HI_deaths,mar21_HI_deaths,total_data_HI_deaths)


#Check null hypothesis of GA confirmed cases for feb and mar 21 using T-TEST  
print("")
print("1 sample T-test for GA confirmed cases")         
Ttest_1sample(feb21_GA_cases,mar21_GA_cases)


#Check null hypothesis of HI confirmed cases for feb and mar 21  using T-TEST   
print("")
print("1 sample T-test for HI confirmed cases")         
Ttest_1sample(feb21_HI_cases,mar21_HI_cases)

#Check null hypothesis of GA deaths for feb and mar 21 using T-TEST  
print("")   
print("1 sample T-test for GA deaths")         
Ttest_1sample(feb21_GA_deaths,mar21_GA_deaths)


#Check null hypothesis of HI deaths for feb and mar 21 using T-TEST  
print("")  
print("1 sample T-test for HI deaths")         
Ttest_1sample(feb21_HI_deaths,mar21_HI_deaths)


#Check null hypothesis of GA confirmed cases for feb and mar 21 using 2 sampele WALD'S TEST 
print("")   
print("Wald's 2 sample test for GA confirmed cases")         
waldsTest_2sample(feb21_GA_cases,mar21_GA_cases)


#Check null hypothesis of HI confirmed cases for feb and mar 21 using 2 sampele WALD'S TEST  
print("")
print("Wald's 2 sample test for HI confirmed cases")         
waldsTest_2sample(feb21_HI_cases,mar21_HI_cases)

#Check null hypothesis of GA deaths for feb and mar 21 using 2 sampele WALD'S TEST  
print("")   
print("Wald's 2 sample test for GA deaths")         
waldsTest_2sample(feb21_GA_deaths,mar21_GA_deaths)


#Check null hypothesis of HI deaths for feb and mar 21 using 2 sampele WALD'S TEST  
print("")  
print("Wald's 2 sample test for HI deaths")         
waldsTest_2sample(feb21_HI_deaths,mar21_HI_deaths)



#Check null hypothesis of GA confirmed cases for feb and mar 21 using 2 sample unpaired T-TEST  
print("")
print("2 sample unpaired T-TEST for GA confirmed cases")         
Ttest_2sample(feb21_GA_cases,mar21_GA_cases)


#Check null hypothesis of HI confirmed cases for feb and mar 21 using 2 sample unpaired T-TEST   
print("")
print("2 sample unpaired T-TEST for HI confirmed cases")         
Ttest_2sample(feb21_HI_cases,mar21_HI_cases)

#Check null hypothesis of GA deaths for feb and mar 21 using 2 sample unpaired T-TEST  
print("")   
print("2 sample unpaired T-TEST for GA deaths")         
Ttest_2sample(feb21_GA_deaths,mar21_GA_deaths)


#Check null hypothesis of HI deaths for feb and mar 21 using 2 sample unpaired T-TEST  
print("")  
print("2 sample unpaired T-TEST for HI deaths")         
Ttest_2sample(feb21_HI_deaths,mar21_HI_deaths)