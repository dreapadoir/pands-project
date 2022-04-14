# pands-project

## Approach
My research on the iris dataset showed its main use is in machine learning. It is a famous dataset generated in the 1930s and used in countless papers, initially in for classification algorithms and more recently for machine learning. As such, it is available from a wide variety of sources. The most straightforward are Comma Separated Variable files but for this project I elected to import the dataset as a pandas dataframe. This will allow me to use a broad spectrum of data analysis tools available through pandas to analyse and report on the data.

Initially, I intend to use pandas functions to generate summary statistics that describe the dataset. Then I will use pyplot to visualise those statistics. When the visualisations are complete that will give me an indication of what attributes will help to classify a particular instance. I then will define a function that takes in dimensions from the user for the four attributes, and insofar as possible, classify the user's data as a particular variety.

I will use a Jupyter notebook called project.ipynb to present the project and keep a separate file with all the code call irisdataset.py. When referring to line numbers, it is irisdataset.py that is being referred to.

### References

## Initial Coding
The first code I've written imports the dataset from the scikit-learn module. From research on scikit-learn.org, I found a method to import the iris dataset as a pandas dataframe. Line 4 imports the dataset in a dictionary like object, with several attributes. If the parameter asframe is set to True, the dataset is import as a dataframe.

The variety names are stored in a separate attribute called target_names, which is a Series. In the dataframe, irisData, the varieties are referred to by their index in the target_names series. The target column in the dataframe therefore contains only 0, 1 and 2.

### References
(https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)