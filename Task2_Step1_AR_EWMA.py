#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 01:23:55 2021

@author: mayankjain
"""

import pandas as pd
import numpy as np


def calculateErrors(original_data,predicted_data):
    Abs_Percent_Error = 0
    Sum_squared_error = 0
    n = len(original_data)
#Apply mse and mape formulas
    for i in range(n):
        Sum_squared_error += (original_data[i]-predicted_data[i])**2
        # Adding check to avoid divide by zero error
        if original_data[i] != 0:
            Abs_Percent_Error += (abs(original_data[i]-predicted_data[i])*100)/original_data[i]
        
    Mean_Squared_Error = Sum_squared_error/n
    Mean_Abs_Percent_Error = Abs_Percent_Error/n
    
    return Mean_Squared_Error,Mean_Abs_Percent_Error


def calculateAR(y_variable,ar_val,day):

#Fetching August data for the particular column   
    data = pd.read_csv('cleaned_data.csv')
    y_mat_train = []
    for index, row in data.iterrows():
        if(row['Date'][6] == '8'):
            y_mat_train.append(row[y_variable])

    
    y_mat = [] 
    y_mat = y_mat_train
    
    duration = 21+day
    rows = duration - ar_val
    cols = ar_val + 1
    x_mat_train = []

# Code to form X matrix which will be used in calculating coefficients
    for i in range(rows):
        row = []
        for j in range(cols):
            if j == 0:
                row.append(1)    
            else:
                row.append(y_mat[j-1+i])

        x_mat_train.append(row)


# Code to calculate Beta coefficients (Reusing out previous assignment code for multiple regression question)
    x_mat_train = np.array(x_mat_train, dtype = None)
    x_mat_train_transpose = x_mat_train.T    
    product_of_X_and_X_transpose = np.matmul(x_mat_train_transpose,x_mat_train)    
    df_inv = np.linalg.inv(product_of_X_and_X_transpose)    
    invmulx_transpose = np.matmul(df_inv,x_mat_train_transpose)    
    coeff = np.matmul(invmulx_transpose,y_mat[ar_val:duration])
    
    #print("coeff: ",coeff)

    
    y_estimated_test = []
    
    for val in y_mat:
        y_estimated_test.append(val)
        
# y_est_val will the estimated value for that day using the coefficients calculated above
    y_est_val = coeff[0]
    for j in  range(ar_val):
        y_est_val += coeff[j+1] * y_estimated_test[duration-j-1]
        duration += 1
    
# return original value of that day and the estimated value of that day to calculate errors next
    return y_mat[21+day],y_est_val


def calculateEWMA(y_variable,alpha):
#Fetching August data for the particular column      
    data = pd.read_csv('cleaned_data.csv')
    y_mat_train = []
    for index, row in data.iterrows():
        if(row['Date'][6] == '8'):
            y_mat_train.append(row[y_variable])

    
    y_mat = [] 
    y_mat = y_mat_train
    
    y_estimated_test = []
    temp = []
    for val in y_mat_train:
        temp.append(val)
    
    duration = 21
    
#Applying ewma series formula as explained in class lectures
    for i in range(7):
        moving_avg = 0
        for j in range (duration+i):
            moving_avg += pow((1-alpha),j) * temp[duration-j-1+i]

            
        y_estimated_test.append(alpha*moving_avg)

#Returning original values for last week of august and predicted values to calculate errors
    return y_mat[21:28],y_estimated_test

    

#Test AR and EWMA code for confirmed cases and deaths of both states

y_variable = 'daily_GA_confirmed'

predicted_cases = []
original_vals = []
# Running a loop to get prediction of 7 days using AR techninque
#For each day we are calculating new coefficients and then predicting
for i in range(7):
    orig_val,predicted_val = calculateAR(y_variable,3,i)
    original_vals.append(orig_val)
    predicted_cases.append(predicted_val)
print("")
print("Predicted confirmed cases for GA with AR = 3: ", predicted_cases)

Mean_Squared_Error,Mean_Abs_Percent_Error = calculateErrors(original_vals,predicted_cases)
print("Mean_Squared_Error: ",Mean_Squared_Error," Mean Absolute Percent Error ",Mean_Abs_Percent_Error)

print("")

predicted_cases = []
original_vals = []
for i in range(7):
    orig_val,predicted_val = calculateAR(y_variable,5,i)
    original_vals.append(orig_val)
    predicted_cases.append(predicted_val)
 
print("Predicted confirmed cases for GA with AR = 5: ",predicted_cases)
Mean_Squared_Error,Mean_Abs_Percent_Error = calculateErrors(original_vals,predicted_cases)
print("Mean_Squared_Error: ",Mean_Squared_Error," Mean Absolute Percent Error ",Mean_Abs_Percent_Error)

y_variable = 'daily_GA_deaths'
print("")

predicted_cases = []
original_vals = []
for i in range(7):
    orig_val,predicted_val = calculateAR(y_variable,3,i)
    original_vals.append(orig_val)
    predicted_cases.append(predicted_val)

print("Predicted deaths for GA with AR = 3: ",predicted_cases)

Mean_Squared_Error,Mean_Abs_Percent_Error = calculateErrors(original_vals,predicted_cases)
print("Mean_Squared_Error: ",Mean_Squared_Error," Mean Absolute Percent Error ",Mean_Abs_Percent_Error)

print("")

predicted_cases = []
original_vals = []
for i in range(7):
    orig_val,predicted_val = calculateAR(y_variable,5,i)
    original_vals.append(orig_val)
    predicted_cases.append(predicted_val)

print("Predicted deaths for GA with AR = 5: ",predicted_cases) 

Mean_Squared_Error,Mean_Abs_Percent_Error = calculateErrors(original_vals,predicted_cases)
print("Mean_Squared_Error: ",Mean_Squared_Error," Mean Absolute Percent Error ",Mean_Abs_Percent_Error)


y_variable = 'daily_GA_confirmed'
print("")
orig_val, predicted_val = calculateEWMA(y_variable,0.5)
print("Predicted confirmed cases for GA with EWMA alpha = 0.5: ",predicted_val)
Mean_Squared_Error,Mean_Abs_Percent_Error = calculateErrors(original_vals,predicted_cases)
print("Mean_Squared_Error: ",Mean_Squared_Error," Mean Absolute Percent Error ",Mean_Abs_Percent_Error)

print("")
orig_val, predicted_val = calculateEWMA(y_variable,0.8)
print("Predicted confirmed cases for GA with EWMA alpha = 0.8: ",predicted_val)
Mean_Squared_Error,Mean_Abs_Percent_Error = calculateErrors(original_vals,predicted_cases)
print("Mean_Squared_Error: ",Mean_Squared_Error," Mean Absolute Percent Error ",Mean_Abs_Percent_Error)


y_variable = 'daily_GA_deaths'
print("")
orig_val, predicted_val = calculateEWMA(y_variable,0.5)
print("Predicted deaths for GA with EWMA alpha = 0.5: ",predicted_val)
Mean_Squared_Error,Mean_Abs_Percent_Error = calculateErrors(original_vals,predicted_cases)
print("Mean_Squared_Error: ",Mean_Squared_Error," Mean Absolute Percent Error ",Mean_Abs_Percent_Error)

print("")
orig_val, predicted_val = calculateEWMA(y_variable,0.8)
print("Predicted deaths for GA with EWMA alpha = 0.8: ",predicted_val)
Mean_Squared_Error,Mean_Abs_Percent_Error = calculateErrors(original_vals,predicted_cases)
print("Mean_Squared_Error: ",Mean_Squared_Error," Mean Absolute Percent Error ",Mean_Abs_Percent_Error)

y_variable = 'daily_HI_confirmed'

predicted_cases = []
original_vals = []
for i in range(7):
    orig_val,predicted_val = calculateAR(y_variable,3,i)
    original_vals.append(orig_val)
    predicted_cases.append(predicted_val)
print("") 
print("Predicted confirmed cases for HI with AR = 3: ",predicted_cases)
Mean_Squared_Error,Mean_Abs_Percent_Error = calculateErrors(original_vals,predicted_cases)
print("Mean_Squared_Error: ",Mean_Squared_Error," Mean Absolute Percent Error ",Mean_Abs_Percent_Error)

print("")

predicted_cases = []
original_vals = []
for i in range(7):
    orig_val,predicted_val = calculateAR(y_variable,5,i)
    original_vals.append(orig_val)
    predicted_cases.append(predicted_val)
 
print("Predicted confirmed cases for HI with AR = 5: ",predicted_cases)
Mean_Squared_Error,Mean_Abs_Percent_Error = calculateErrors(original_vals,predicted_cases)
print("Mean_Squared_Error: ",Mean_Squared_Error," Mean Absolute Percent Error ",Mean_Abs_Percent_Error)

y_variable = 'daily_HI_deaths'
print("")

predicted_cases = []
original_vals = []
for i in range(7):
    orig_val,predicted_val = calculateAR(y_variable,3,i)
    original_vals.append(orig_val)
    predicted_cases.append(predicted_val)
 
print("Predicted deaths for HI with AR = 3: ",predicted_cases)
Mean_Squared_Error,Mean_Abs_Percent_Error = calculateErrors(original_vals,predicted_cases)
print("Mean_Squared_Error: ",Mean_Squared_Error," Mean Absolute Percent Error ",Mean_Abs_Percent_Error)

print("")

predicted_cases = []
original_vals = []
for i in range(7):
    orig_val,predicted_val = calculateAR(y_variable,5,i)
    original_vals.append(orig_val)
    predicted_cases.append(predicted_val)
 
print("Predicted deaths for HI with AR = 5: ",predicted_cases)
Mean_Squared_Error,Mean_Abs_Percent_Error = calculateErrors(original_vals,predicted_cases)
print("Mean_Squared_Error: ",Mean_Squared_Error," Mean Absolute Percent Error ",Mean_Abs_Percent_Error)


y_variable = 'daily_HI_confirmed'
print("")
orig_val, predicted_val = calculateEWMA(y_variable,0.5)
print("Predicted confirmed cases for HI with EWMA alpha = 0.5: ",predicted_val)
Mean_Squared_Error,Mean_Abs_Percent_Error = calculateErrors(orig_val,predicted_val)
print("Mean_Squared_Error: ",Mean_Squared_Error," Mean Absolute Percent Error ",Mean_Abs_Percent_Error)

print("")
orig_val, predicted_val = calculateEWMA(y_variable,0.8)
print("Predicted confirmed cases for HI with EWMA alpha = 0.8: ",predicted_val)
Mean_Squared_Error,Mean_Abs_Percent_Error = calculateErrors(orig_val,predicted_val)
print("Mean_Squared_Error: ",Mean_Squared_Error," Mean Absolute Percent Error ",Mean_Abs_Percent_Error)


y_variable = 'daily_HI_deaths'
print("")
orig_val, predicted_val = calculateEWMA(y_variable,0.5)
print("Predicted deaths for HI with EWMA alpha = 0.5: ",predicted_val)
Mean_Squared_Error,Mean_Abs_Percent_Error = calculateErrors(orig_val,predicted_val)
print("Mean_Squared_Error: ",Mean_Squared_Error," Mean Absolute Percent Error ",Mean_Abs_Percent_Error)

print("")
orig_val, predicted_val = calculateEWMA(y_variable,0.8)
print("Predicted deaths for HI with EWMA alpha = 0.8: ",predicted_val)
Mean_Squared_Error,Mean_Abs_Percent_Error = calculateErrors(orig_val,predicted_val)
print("Mean_Squared_Error: ",Mean_Squared_Error," Mean Absolute Percent Error ",Mean_Abs_Percent_Error)

