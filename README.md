# pands-project

## Requirements
This code was written on MS Visual Code (version 1.64.0) using Python 3.9.7. 
Running code contained in this repository will require Python 3.9.7 or higher.

The latest release of the Anaconda distribution will be sufficient (available [here](https://www.anaconda.com/products/individual)).

The following libraries will be required:
- pandas
- pyplot
- sklearn.datasets


## Approach
My research on the iris dataset showed its main use is in machine learning. It is a famous dataset generated in the 1930s and used in countless papers, initially in for classification algorithms and more recently for machine learning. As such, it is available from a wide variety of sources. The most straightforward are Comma Separated Variable files but for this project I elected to import the dataset as a pandas dataframe. This will allow me to use a broad spectrum of data analysis tools available through pandas to analyse and report on the data.

Initially, I intend to use pandas functions to generate summary statistics that describe the dataset. Then I will use pyplot to visualise those statistics. When the visualisations are complete that will give me an indication of what attributes will help to classify a particular instance. I then will define a function that takes in dimensions from the user for the four attributes, and insofar as possible, classify the user's data as a particular variety.

### References
(https://archive.ics.uci.edu/ml/datasets/iris)

## The dataset
The actual data was collected by a botanist named Edgar Anderson in 1930. Three different but related species of iris were studied, iris setosa, iris virginica and iris versicolor. Anderson measured four attributes of each species; the width and length of the petals and the width and length of the sepal (the green outer layer of buds that remains below the petals on the stem after flowering). Anderson's data was used by the statistican R. A Fisher to test classification models. A more recent use case has been the application to machine learning. 

### References
(https://en.wikipedia.org/wiki/Iris_flower_data_set)

## Initial Coding
The first code I've written imports the dataset from the scikit-learn module. From research on scikit-learn.org, I found a method to import the iris dataset as a pandas dataframe. Line 4 imports the dataset in a dictionary like object, with several attributes. If the parameter asframe is set to True, the dataset is imported as a dataframe.

The variety names are stored in a separate attribute called target_names, which is a Series. In the dataframe, irisData, the varieties are referred to by their index in the target_names series. The target column in the dataframe therefore contains only 0, 1 and 2.

### References
(https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)

## Code
The main summary statistics for each variable are outputted to the file [summary.txt](summary.txt). The first code to set this file up runs on line 12. The variety names are output first, along with their corresponding code. The names are pulled from the target_names list using their index. The argument "w" is passed to the open function so that this code can write to the summary file. "w" also means it will overwrite the contents of the file each time the code is run, rather than appending to it.

The remainder of the code that writes to the summary file begins on line 23. This code takes the dataframe and aggregates the four attributes, grouping them by the variety (target). The min, max, median, mean and standard deviation of each attribute are displayed in the summary file. In this block of code the argument "a" is passed to the open function so this data is appended to the output from line 12, rather than overwriting it.

On line 15 and 17, the structure of the data is looked at. Using the head function, line 15 prints out the first 5 lines of the dataframe to give an overall idea of how the structure looks, column names etc. Line 17 uses the count function on the target column to identify how many instances are in the dataset. Once the column names and the size of the dataset have been established, more detailed examination of the data can begin. The data can now be sliced by column and row to look at individual attributes and look at the varieties separately.

Line 20 uses the describe function to list all the availabe summary statistics for the dataset as whole. While interesting, they don't contribute much to help classify the varieties which will be the main focus of this project. For the remainder of this study, the data will be looked at on a variety scale. From our initial investigations we can establish a slicing pattern of lines 0 to 49 for setosa, 50 to 99 for versicolor and 100 to 149 for virginica. We will use these patterns, written as slicing statements, to build a model to help classify the varieties.

Two methods of visualising the data will help with establishing the structure of the model, boxplots and scatterplots. The boxplots plot the range of each variety for each attribute on a single plot, allowing an easy comparision between varieties. Examining the resulting plots, particularly the petal length and petal width plots, make it clear the that the setosa variety dimensions are disjoint from those of the other two. There is clear separation between the setosa cluster and the larger versicolor/virginica cluster. It follows that an analysis of the petal lengths and widths should immediately identify if a sample is a setosa or not.

The scatterplots plot pairs of attributes against each other. The resulting data points are coloured by variety to visualise the interaction between them. Again we see that setosa data points are generally distinct from any other points but there is a degree of mixing between virginica and versicolor. This implies that it will not be possible to to make a simple distinction between those varieties based on one or two dimensions.

The model to classify an instance of iris from user submitted data is started on line 113.

To differentiate between varieties, a 6 sigma confidence interval can be used. In this case sigma refers to the standard deviation of a population. Statistically, 99.99966% of all values of a population are expected to fall within 6 standard deviations of the mean.

The first part of the model looks at the petal widths and length are examined. If

For the purposes of this model, each data point entered by the user will be compared to the 6 six sigma confidence intervals for that attribute of both versicolor and virginica. If the datapoint is within the interval, a 1 is added to the empty list

### References
(https://sixsigmastudyguide.com/confidence-intervals/)
(https://en.wikipedia.org/wiki/Six_Sigma)