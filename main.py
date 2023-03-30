import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3" #label- what we are trying to get

X = np.array(data.drop([predict], 1))  #training data used to predict new value
y = np.array(data[predict]) #all the labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.5)

best = 0
'''
for _ in range(30):
    # all labels split up into four different arrays (50% of our data)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.5)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("The Accuracy of this test:",str(acc*100),"%") #prints the accuracy of our predictions
    if acc>best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
'''
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

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


