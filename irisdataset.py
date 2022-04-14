#program to analyse the iris dataset
#Author: David Higgins

from sklearn.datasets import load_iris          #loads the modules necessary for this code
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats

data = load_iris(return_X_y = False, as_frame = True)                 #loads the iris dataset as a dataframe

irisData = data.frame                           #creates a variable containing the dataframe

print("Target codes:\n0 =", data.target_names[0], "\n1 =", data.target_names[1], "\n2 =", data.target_names[2]) #prints out the variety names equivalent to the target codes 

print(irisData.head())                          #shows the first 5 lines of the dataframe to illustrate the structure

print("There are {} samples in the dataset".format(irisData['target'].count()))     #counts the samples in the dataset by counting each entry in the target column

print(irisData.describe())                      #describes the dataset as a whole. It gives summary statistics for each attribute without considering the separate varieties.

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
sepLen = [                                #creates a list containing the sepal lengths of each variety
    irisData.iloc[:50]['sepal length (cm)'],irisData.iloc[50:100]['sepal length (cm)'],
    irisData.iloc[100:150]['sepal length (cm)']
]

sepWid = [                                #creates a list containing the sepal widths of each variety
    irisData.iloc[:50]['sepal width (cm)'],irisData.iloc[50:100]['sepal width (cm)'],
    irisData.iloc[100:150]['sepal width (cm)']
]

petLen = [                                #creates a list containing the petal lengths of each variety
    irisData.iloc[:50]['petal length (cm)'],irisData.iloc[50:100]['petal length (cm)'],
    irisData.iloc[100:150]['petal length (cm)']
]

petWid = [                                #creates a list containing the petal widths of each variety
    irisData.iloc[:50]['petal width (cm)'],irisData.iloc[50:100]['petal width (cm)'],
    irisData.iloc[100:150]['petal width (cm)']
]

plt.boxplot(sepLen, labels = labels)      
plt.ylabel("sepal length in cm")
plt.title("Sepal Length")
plt.show()

plt.boxplot(sepWid, labels = labels)      
plt.ylabel("sepal width in cm")
plt.title("Sepal Width")
plt.show()

plt.boxplot(petLen, labels = labels)      
plt.ylabel("petal length in cm")
plt.title("petal Length")
plt.show()

plt.boxplot(petWid, labels = labels)      
plt.ylabel("petal width in cm")
plt.title("Petal Width")
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

intervals = [range(0, 50), range(50, 100), range(100,150)]          #creates slices equivalent to the locations of the three varieties of iris

for interval in intervals:                     #calls the function in a for loop to print out the 95% confidence intervals of all 3 varieties
    print("Petal length confidence intervals:")
    print(data.target_names[(min(interval))//50])
    mean_confidence_interval(irisData['petal length (cm)'][interval])
        
sepallength = input("Enter sepal length: ")
sepalwidth = input("Enter sepal width: ")
petallength = input("Enter petal length: ")
petalwidth = input("Enter petal width: ")

