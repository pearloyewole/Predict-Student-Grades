import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["sex", "G1", "G2", "G3", "studytime", "failures", "absences"]]
data = pd.get_dummies(data, columns=["sex"])

predict = "G3" #label- what we are trying to get

X = np.array(data.drop([predict], 1))  #training data used to predict new value
y = np.array(data[predict]) #all the labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.5)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

print("Coefficients", linear.coef_) #the larger the coefficients the greater the weight it has
print("Intercept", linear.intercept_)

predictions = linear.predict(x_test)

for x in range (len(predictions)):
    print("Grade Predicted:",predictions[x], x_test[x], "Actual Grade: ",y_test[x])

p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"]) # x and y axis of graph
pyplot.ylabel("Final Grade") #title of graph
pyplot.show()
