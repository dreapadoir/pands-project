# pands-project

## Approach
My research on the iris dataset showed its main use is in machine learning. It is a famous dataset generated in the 1930s and used in countless papers, initially in for classification algorithms and more recently for machine learning. As such, it is available from a wide variety of sources. The most straightforward are Comma Separated Variable files but for this project I elected to import the dataset as a pandas dataframe. This will allow me to use a broad spectrum of data analysis tools available through pandas to analyse and report on the data.

Initially, I intend to use pandas functions to generate summary statistics that describe the dataset. Then I will use pyplot to visualise those statistics. When the visualisations are complete that will give me an indication of what attributes will help to classify a particular instance. I then will define a function that takes in dimensions from the user for the four attributes, and insofar as possible, classify the user's data as a particular variety.

### References

## The dataset
The actual data was collected by a botanist named Edgar Anderson in 1930. Three different but related species of iris were studied, iris setosa, iris virginica and iris versicolor. Anderson measured four attributes of each species; the width and length of the petals and the width and length of the sepal (the green outer layer of buds that remains below the petals on the stem after flowering). Anderson's data was used by the statistican R. A Fisher to test classification models. A more recent use case has been the application to machine learning. 

### References
(https://en.wikipedia.org/wiki/Iris_flower_data_set)

## Initial Coding
The first code I've written imports the dataset from the scikit-learn module. From research on scikit-learn.org, I found a method to import the iris dataset as a pandas dataframe. Line 4 imports the dataset in a dictionary like object, with several attributes. If the parameter asframe is set to True, the dataset is imported as a dataframe.

The variety names are stored in a separate attribute called target_names, which is a Series. In the dataframe, irisData, the varieties are referred to by their index in the target_names series. The target column in the dataframe therefore contains only 0, 1 and 2.

### References
(https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)