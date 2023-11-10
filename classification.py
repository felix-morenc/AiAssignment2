import streamlit as st
from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# fetch dataset 
rice_cammeo_and_osmancik = fetch_ucirepo(id=545) 
  
# data (as pandas dataframes) 
X = rice_cammeo_and_osmancik.data.features 
y = rice_cammeo_and_osmancik.data.targets


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


st.title('ML Algorithms Testing')

st.header("Introduction")

st.write('When talking about machine learning there are a number of different algorithms and strategies that can be used. From Random Forest to Naive Bayes, these algorithms all have their own specific strengths and weaknesses, usually dependant on the data set that is being worked with. For this task I have chosen to demonstrate the use of 3 different classification oriented algorithms to perform classification. Through the use of the Rice classification dataset from the UCI database, I aim to analyse the manner in which these various algorithms function, and why certain ones are more suited to this dataset resulting in them performing with a greater accuracy than others.')

st.header("Taking a deeper look into the data (EDA)")

st.write("Before working with this dataset deeper, we need to analayse and make sure it meets all our requirements such as: 5 different classifiers, more than 100 data points, and having no empty values")

st.write("Below you can see the full dataframe of our Rice data that we pulled from the UCI database")

st.write(rice_cammeo_and_osmancik.data.original)

st.write("Taking a deep look into the data we see that we have over 5 different classifiers, specifically 7. Additionally the dataset is populated with 3810 instances, well over out minimum.")

st.write("To check if there's any missing data, we check the dataset's metadata, specifically the has_missing_values point:")

st.write(rice_cammeo_and_osmancik.metadata.has_missing_values)

st.write("As we are returned no we know that this dataset is also complete, meaning it meets all of our requirements")

st.header("Importing and Sorting Data")

code =  '''
  
# data (as pandas dataframes) 
X = rice_cammeo_and_osmancik.data.features 
y = rice_cammeo_and_osmancik.data.targets

#splitting data into training and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(f"No. of training examples: {X_train.shape[0]}")
print(f"No. of testing examples: {X_test.shape[0]}")'''

st.write("The code below is what we use to split the data into training and test datasets")

st.code(code, language='python')

st.write(f"No. of training examples: {X_train.shape[0]}")
st.write(f"No. of testing examples: {X_test.shape[0]}")

st.header("Algorithm 1: Random Forest")

codeForest = '''#Running the random forest using our training data
rf = RandomForestClassifier()
rf.fit(X_train, y_train.values.ravel())

#Testing our model using our test data
y_pred = rf.predict(X_test)

#calculate our accuracy score
accuracy = accuracy_score(y_test, y_pred)
st.write("Accuracy:", accuracy)

#generate confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)

st.write(cm)


'''
st.write("Below is the code used to run the Random Forest Model as well as the accuracy and confusion matrix it generates")
st.code(codeForest, language='python')
#Running the random forest using our training data
rf = RandomForestClassifier()
rf.fit(X_train, y_train.values.ravel())

#Testing our model using our test data
y_pred = rf.predict(X_test)

#calculate our accuracy score
accuracy = accuracy_score(y_test, y_pred)
st.write("Accuracy:", accuracy)

#generate confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)

st.write("Confusion Matrix")
st.write(cm)

st.header("Algorithm 2: Support Vector Machines")

codeSvm = '''#Creating classifier for SVM
clf = svm.SVC(kernel='linear')

#Train model with training data
clf.fit(X_train, y_train.values.ravel())

#Testing model using acquired test data
y_pred = clf.predict(X_test)

#Calculate and print out accuracy of model
st.write("Accuracy:", metrics.accuracy_score(y_test, y_pred))

cm = metrics.confusion_matrix(y_test, y_pred)
st.write(cm)
'''

st.write("Below is the code used to run the Support Vector Machines as well as the accuracy and confusion matrix it generates")

st.code(codeSvm, language='python')

#Creating classifier for SVM
clf = svm.SVC(kernel='linear')

#Train model with training data
clf.fit(X_train, y_train.values.ravel())

#Testing model using acquired test data
y_pred = clf.predict(X_test)

#Calculate and print out accuracy of model
st.write("Accuracy:", metrics.accuracy_score(y_test, y_pred))

cm2 = metrics.confusion_matrix(y_test, y_pred)

st.write(cm2)

st.header("Algorithm 3: Gaussian Naive Bayes")

codeGnb = '''#Creating model
nbmodel = GaussianNB()

#Using training data to train the model
nbmodel.fit(X_train, y_train.values.ravel())

#using test data to predict outcome
y_pred = nbmodel.predict(X_test)

#calculating accuracy of model using test data
accuracy = accuracy_score(y_pred, y_test)

st.write("Accuracy:", accuracy)

cm = metrics.confusion_matrix(y_test, y_pred)

st.write(cm)
'''

st.write("Below is the code used to run the Gaussian Naive Bayes as well as the accuracy and confusion matrix it generates")

st.code(codeGnb, language='python')

#Creating model
nbmodel = GaussianNB()

#Using training data to train the model
nbmodel.fit(X_train, y_train.values.ravel())

#using test data to predict outcome
y_pred = nbmodel.predict(X_test)

#calculating accuracy of model using test data
accuracy2 = accuracy_score(y_pred, y_test)

st.write("Accuracy:", accuracy2)

cm3 = metrics.confusion_matrix(y_test, y_pred)

st.write(cm3)

st.header("Conclusion")

st.write('''Based on the results that we have achieved there are a number of things we have learned, as well as many things we need to take into consideration when applying each of these algorithms, that could have varying levels importance based on the data that is being worked on

To begin, when looking at the results from all 3 algorithms, we never achieved an accuaracy of lower than 0.9 or 90%. This means we're dealing with a rather consistent and clear dataset where the types of rice being dealt with can be deemed to be rather distinct, making it easier to get a quite accurate prediction from our trained models.

Starting with the "Hello World" of classification, the Random Forest algorithm, we were able to achieve an accuracy of 0.92, which was the second highest accuracy, however important to consider the margin between it and the highest accuracy achieved with the SVM model was within a margin of error that could quite easily change based on the splitting of training and testing data. Only inaccurately classifying 60/762 of our test data points is a commendable performance and leaves us with a rather reliable model. Computationally it was the median performer of our 3 techiniques, taking 3.3 seconds to train and test our model. I'd say as a whole this is a model that equally trades off accuracy and computational power to get us a perfectly satisfacotry result

Support Vector Machines, otherwise known as SVM, achieved the highest accuracy of our 3 models, scoring a 0.937 or 93.7%. This means only 48/762 of the point is our test set were misclassified, meaning that if absolute accuracy is the most important feature needed when running your classificatiion algorithms, this will likely be one of the best options avaiable. This does however come at a computational cost which could be very important to consider when working with gargantuan data sets. Taking a whole 7.7 seconds to train and test the model, it took over double the time of the second slowest performer. Thus to conclude if the goal is to achieve accuracy never mind the computational cost, SVM appears to be the best algorithm to use out of our 3 test candidates

Finally, the Gaussian Naive Bayes algorithm achieved the lowest accuracy score we saw from the 3 candidates today, only scoring 0.909 or 90.9%. This means that out of our 762 test data points, it inaccurately classified 69 of them, which when dealing with more critical classification issues, it could be dangerous and result in quite a lot of misclassifications. The huge advantage however was the sheer speed of the computation that this model achieved. At only 0.3 seconds to train the model and test the data, it's a blisteringly quick model and would perform excellently if computational resources are hard to come by

All in all, all three of the models have their strengths and weaknesses, which model to use will eventually come down to what the user values most out of their model: accuracy, speed or a middle ground of both factors''')







