#program to analyse the iris dataset
#Author: David Higgins

from sklearn.datasets import load_iris

data = load_iris(as_frame=True)

irisData = data.frame

setosaData = irisData.iloc[:50]
versicolorData = irisData.iloc[50:100]
virginicaData = irisData.iloc[100:150]

avg = setosaData['sepal length (cm)'].sum(axis = 0)/len(setosaData['sepal length (cm)'])
avg1 = versicolorData['sepal length (cm)'].sum(axis = 0)/len(versicolorData['sepal length (cm)'])
avg2 = virginicaData['sepal length (cm)'].sum(axis = 0)/len(virginicaData['sepal length (cm)'])

print('Average sepal lengths are:', round(avg, 3), round(avg1, 3), round(avg2, 3))