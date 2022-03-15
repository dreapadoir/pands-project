#program to analyse the iris dataset
#Author: David Higgins

from sklearn.datasets import load_iris

data = load_iris(as_frame=True)

irisdf = data.frame