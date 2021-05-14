import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy.stats import gamma
import scipy

df = pd.read_csv("cleaned_data.csv")
    
#filter june month data
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Date'] >= datetime.datetime(2020, 6, 1)]

#sum up respective stats for calculation(cases = sum of cases , deaths = sum of deaths)
cases = df["daily_GA_confirmed"] + df["daily_HI_confirmed"]
deaths = df["daily_GA_deaths"] + df["daily_HI_deaths"]

#do bayesian inference for 4 times assuming prior(for 4 weeks)
def do_bayessian(data, data_name):
    for i in range(4):
        week_to_calculate = i + 1
        end = 28 + (week_to_calculate * 7)
        alpha = sum(data[: end]) + 1
        beta = len(data[: end]) + ( len(data[: end]) / sum(data[: end]) )
        scale = 1 / beta
        
        line_space_start = scipy.stats.gamma.ppf(0.001, alpha, scale=scale)
        line_space_end = scipy.stats.gamma.ppf(0.999, alpha, scale=scale)
        x = np.linspace(line_space_start, line_space_end, 1000)
        
        print('Week :', str(week_to_calculate))
        print('alpha :', alpha, 'MAP :', alpha/beta)
        plt.title('Posterior gamma distributions for ' + data_name)
        plt.gca().set(xlabel = 'Number of ' + data_name, ylabel = 'PDF of Gamma distributions')
        plt.plot(x, gamma.pdf(x, alpha, scale=1/beta))
    plt.show()
    
do_bayessian(cases, 'cases')
do_bayessian(deaths, 'deaths')
