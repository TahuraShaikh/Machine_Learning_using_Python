import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle


data = pd.read_csv("student-mat.csv ", sep=";")

data=data[["G1","G2","G3","studytime","failures","absences"]]             #features

predict="G3"                                                              #label

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.1)       #dividing test and train data

linear=linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc= linear.score(x_test,y_test)
print(acc)

'''predictions=linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test)'''

# saving our model
with open("studentgrades.pickle", "wb") as f:
    pickle.dump(linear, f)
