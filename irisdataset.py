#program to analyse the iris dataset
#Author: David Higgins

from sklearn.datasets import load_iris          #loads the modules necessary for this code
from matplotlib import pyplot as plt
import pandas as pd

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