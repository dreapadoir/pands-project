#program to analyse the iris dataset
#Author: David Higgins

from sklearn.datasets import load_iris          #loads the modules necessary for this code
from matplotlib import pyplot as plt
import pandas as pd

data = load_iris(return_X_y = False, as_frame = True)                 #loads the iris dataset as a dataframe

irisData = data.frame                           #creates a variable containing the dataframe

print("Target codes:\n0 =", data.target_names[0], "\n1 =", data.target_names[1], "\n2 =", data.target_names[2], file=open("summary.txt", "w")) #prints out the variety names equivalent to the target codes 

print("\nStructure of the dataframe:")
print("\n",irisData.head())                          #shows the first 5 lines of the dataframe to illustrate the structure

print("\nThere are {} samples in the dataset".format(irisData['target'].count()))     #counts the samples in the dataset by counting each entry in the target column

print("\nSummary statistics for dataset:")
print("\n",irisData.describe())                      #describes the dataset as a whole. It gives summary statistics for each attribute without considering the separate varieties.


print("\n",irisData.groupby('target').agg(           #groups the data by variety (target) and aggregates the summary statistics listed
    {
        'sepal length (cm)': ["min", "max", "mean", "median", "std" ],
        'sepal width (cm)': ["min", "max", "mean", "median", "std"],
        'petal length (cm)': ["min", "max", "mean", "median", "std"],
        'petal width (cm)': ["min", "max", "mean", "median", "std"]
    }

)
, file=open("summary.txt", "a"))           #outputs the summary statistic data from this print statement to a file called summary.txt


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

plt.boxplot(sepLen, labels = labels)      #creates a boxplot for each characteristic grouped by variety
plt.ylabel("sepal length in cm")
plt.title("Sepal Length")
plt.savefig("Sepal Length boxplot.png")
plt.show()


plt.boxplot(sepWid, labels = labels)      
plt.ylabel("sepal width in cm")
plt.title("Sepal Width")
plt.savefig("Sepal Width boxplot.png")
plt.show()


plt.boxplot(petLen, labels = labels)      
plt.ylabel("petal length in cm")
plt.title("petal Length")
plt.savefig("Petal Length boxplot.png")
plt.show()


plt.boxplot(petWid, labels = labels)      
plt.ylabel("petal width in cm")
plt.title("Petal Width")
plt.savefig("Petal Width boxplot.png")
plt.show()

plt.scatter(petLen, petWid, c = data.target)            #creates a scatterplot for each pair of characteristics, colouring them by variety to differentiate each point
plt.title("Plot of petal length vs petal width")
plt.savefig("Plot of petal length vs petal width.png")
plt.show()

plt.scatter(sepLen, sepWid, c = data.target)
plt.title("Plot of sepal length vs sepal width")
plt.savefig("Plot of sepal length vs sepal width.png")
plt.show()

plt.scatter(petLen, sepWid, c = data.target)
plt.title("Plot of petal length vs sepal width")
plt.savefig("Plot of petal length vs sepal width.png")
plt.show()

plt.scatter(sepLen, petWid, c = data.target)
plt.title("Plot of sepal length vs petal width")
plt.savefig("Plot of sepal length vs petal width.png")
plt.show()

plt.scatter(petLen, sepLen, c = data.target)
plt.title("Plot of petal length vs sepal length")
plt.savefig("Plot of petal length vs sepal length.png")
plt.show()

plt.scatter(sepWid, petWid, c = data.target)
plt.title("Plot of sepal width vs petal width")
plt.savefig("Plot of sepal width vs petal width.png")
plt.show()

print("\nStandard deviations of each attribute (to be used to construct six sigma intervals):")
print("\n",irisData.groupby('target').agg(   #finds the standard deviation of the four attributes for each variety of iris
    {
        'sepal length (cm)': ["std"],
        'sepal width (cm)': ["std"],
        'petal length (cm)': ["std"],
        'petal width (cm)': ["std"]
    }
)
)

def inputdata():
    virginicascore = []         #creates two empty lists to hold integers from classification model in next block of text
    versicolorscore = []

    print("\nAlgorithm to classify a particular sample of iris:\n")
    
    sepallength = float(input("Enter sepal length: "))     #accepts iris attribute data from user and stores in a variable
    sepalwidth = float(input("Enter sepal width: "))
    petallength = float(input("Enter petal length: "))
    petalwidth = float(input("Enter petal width: "))

                                                            #this block checks the setosa petal dimensions first as they are the easiest way to differentiate the varieties initially
    if petallength >= (irisData['petal length (cm)'][0:50].mean() - (6 * (irisData['petal length (cm)'][0:50].std()))) and \
    petallength <= (irisData['petal length (cm)'][0:50].mean() + (6 * (irisData['petal length (cm)'][0:50].std()))) and \
    petalwidth >= (irisData['petal width (cm)'][0:50].mean() - (6 * (irisData['petal width (cm)'][0:50].std()))) and \
    petalwidth <= (irisData['petal width (cm)'][0:50].mean() + (6 * (irisData['petal width (cm)'][0:50].std()))):
        print("This sample is iris setosa")

    else:                                           #if the setosa petal dimensions do not identify the variety as a setosa immediately, we then consider the other two varieties. We use six sigma intervals to determine the probability of the users sample being a particular variety
        if petallength >= (irisData['petal length (cm)'][51:100].mean() - (6 * (irisData['petal length (cm)'][51:100].std()))) and \
        petallength <= (irisData['petal length (cm)'][51:100].mean() + (6 * (irisData['petal length (cm)'][51:100].std()))):
            versicolorscore.append(1)

        if petalwidth >= (irisData['petal width (cm)'][51:100].mean() - (6 * (irisData['petal width (cm)'][51:100].std()))) and \
        petalwidth <= (irisData['petal width (cm)'][51:100].mean() + (6 * (irisData['petal width (cm)'][51:100].std()))):
            versicolorscore.append(1)

        if sepallength >= (irisData['sepal length (cm)'][51:100].mean() - (6 * (irisData['sepal length (cm)'][51:100].std()))) and \
        sepallength <= (irisData['sepal length (cm)'][51:100].mean() + (6 * (irisData['sepal length (cm)'][51:100].std()))):
            versicolorscore.append(1)

        if sepalwidth >= (irisData['sepal width (cm)'][51:100].mean() - (6 * (irisData['sepal width (cm)'][51:100].std()))) and \
        sepalwidth <= (irisData['sepal width (cm)'][51:100].mean() + (6 * (irisData['sepal width (cm)'][51:100].std()))):
            versicolorscore.append(1)

        if petallength >= (irisData['petal length (cm)'][101:150].mean() - (6 * (irisData['petal length (cm)'][101:150].std()))) and \
        petallength <= (irisData['petal length (cm)'][101:150].mean() + (6 * (irisData['petal length (cm)'][101:150].std()))):
            virginicascore.append(1)

        if petalwidth >= (irisData['petal width (cm)'][101:150].mean() - (6 * (irisData['petal width (cm)'][101:150].std()))) and \
        petalwidth <= (irisData['petal width (cm)'][101:150].mean() + (6 * (irisData['petal width (cm)'][101:150].std()))):
            virginicascore.append(1)

        if sepallength >= (irisData['sepal length (cm)'][101:150].mean() - (6 * (irisData['sepal length (cm)'][101:150].std()))) and \
        sepallength <= (irisData['sepal length (cm)'][101:150].mean() + (6 * (irisData['sepal length (cm)'][101:150].std()))):
            virginicascore.append(1)

        if sepalwidth >= (irisData['sepal width (cm)'][101:150].mean() - (6 * (irisData['sepal width (cm)'][101:150].std()))) and \
        sepalwidth <= (irisData['sepal width (cm)'][101:150].mean() + (6 * (irisData['sepal width (cm)'][101:150].std()))):
            versicolorscore.append(1)

        if sum(versicolorscore) > sum(virginicascore):      #this block checks the sum of the score lists to see which is bigger and therefore more likely to be the variety of the user's sample
            print("\nThis sample is iris versicolor\n")
        elif sum(versicolorscore) < sum(virginicascore):
            print("\nThis sample is iris virginica\n")
        elif sum(versicolorscore) == sum(virginicascore):
            print("This sample is either iris versicolor or iris virginica but it cannot be differentiated any further using this model\n")
    

numbertest = True           #will be used to manage the while loop in the next block.

while numbertest == True:   #this block will run while numbertest is equal to true
    try:                    #this line runs inputdata function and if successful changes numbertest to false and stops the loop
        inputdata()
        numbertest = False
    except:
        print("\nError: you must enter all data as numbers. Please try again")  #this line prints an error and restarts the inputdata function if a number is not entered by the user for any of the dimensions
