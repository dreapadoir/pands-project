#program to analyse the iris dataset
#Author: David Higgins

from sklearn.datasets import load_iris

data = load_iris(as_frame=True)

irisData = data.frame

setosaData = irisData.iloc[:50]
versicolorData = irisData.iloc[50:100]
virginicaData = irisData.iloc[100:150]