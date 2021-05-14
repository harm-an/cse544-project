import numpy as np

def get_pearson_numerator(data1, mean1, data2, mean2):
    # Sum(data1 - mean(data1))(data2 - mean(data2))
    sum_ = 0
    for i in range(len(data1)):
        val = (data1[i] - mean1) * (data2[i] - mean2)
        sum_ += val
    
    return sum_

def get_pearson_denominator(data1, mean1, data2, mean2):
    # sqrt(sum((data1 - mean(data1))^2) * sum((data2 - mean(data2))^2))
    sum1_ = 0
    for i in range(len(data1)):
        val = pow((data1[i] - mean1), 2)
        sum1_ += val

    sum2_ = 0
    for i in range(len(data2)):
        val = pow((data2[i] - mean2), 2)
        sum2_ += val
    
    return np.sqrt(sum1_ * sum2_) 

def pearson_test(data1, data2):
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)

    numerator = get_pearson_numerator(data1, mean1, data2, mean2)
    denominator = get_pearson_denominator(data1, mean1, data2, mean2)

    pearson_coeff = numerator/denominator
    return pearson_coeff