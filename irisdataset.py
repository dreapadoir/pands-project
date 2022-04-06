#program to analyse the iris dataset
#Author: David Higgins

from sklearn.datasets import load_iris          #loads the modules necessary for this code
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats

data = load_iris(as_frame=True)                 #loads the iris dataset as a dataframe

irisData = data.frame                           #creates a variable containing the dataframe

print("Target codes:\n0 =", data.target_names[0], "\n1 =", data.target_names[1], "\n2 =", data.target_names[2]) #prints out the variety names equivalent to the target codes 

print(irisData.groupby('target').agg(           #groups the data by variety (target) and aggregates the summary statistics listed
    {
        'sepal length (cm)': ["min", "max", "mean" ],
        'sepal width (cm)': ["min", "max", "mean"],
        'petal length (cm)': ["min", "max", "mean"],
        'petal width (cm)': ["min", "max", "mean"]
    }

)
)

labels = data.target_names                  #creates a list of the variety names
sepLenBx = [                                #creates a list containing the sepal lengths of each variety
    irisData.iloc[:50]['sepal length (cm)'],irisData.iloc[50:100]['sepal length (cm)'],
    irisData.iloc[100:150]['sepal length (cm)']
]

plt.boxplot(sepLenBx, labels = labels)      #plots boxplots of each set of data in sepLenBx and assigns the variety names as labels
plt.show()

print(irisData.groupby('target').agg(   #finds the standard deviation of the four attributes for each variety of iris
    {
        'sepal length (cm)': ["std"],
        'sepal width (cm)': ["std"],
        'petal length (cm)': ["std"],
        'petal width (cm)': ["std"]
    }
)
)

def mean_confidence_interval(data, confidence=0.95):      #uses scipy and numpy to create a 95% confidence interval
    a = 1.0 * data
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return print('The 95% confidence interval is ({}, {})'.format(round(m-h, 3), round(m+h, 3)))

intervals = [range(0, 50), range(50, 100), range(100,150)]

for interval in intervals:
        mean_confidence_interval(irisData['petal length (cm)'][interval])