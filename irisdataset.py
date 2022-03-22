#program to analyse the iris dataset
#Author: David Higgins

from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
import pandas as pd

data = load_iris(as_frame=True)

irisData = data.frame

setosaData = irisData.iloc[:50]
versicolorData = irisData.iloc[50:100]
virginicaData = irisData.iloc[100:150]

print("Target codes:\n0 =", data.target_names[0], "\n1 =", data.target_names[1], "\n2 =", data.target_names[2])

print(irisData.groupby('target').agg(
    {
        'sepal length (cm)': ["min", "max", "mean" ],
        'sepal width (cm)': ["min", "max", "mean"],
        'petal length (cm)': ["min", "max", "mean"],
        'petal width (cm)': ["min", "max", "mean"]
    }

)
)

