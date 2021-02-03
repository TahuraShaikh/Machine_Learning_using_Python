# Linear Regresion Model to find mean squared error, weights, intercept :

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes=datasets.load_diabetes()
#diabetes_X=diabetes.data[:,np.newaxis,2]           #gives array of arrays , used for only one value

diabetes_X=diabetes.data
#print(diabetes_X)

#Slicing of data:

#Features:
diabetes_X_train=diabetes_X[:-30]            # last  30 to train
diabetes_X_test=diabetes_X[-30:]             # first 30 to test

# Labels:
diabetes_y_train=diabetes.target[:-30]
diabetes_y_test=diabetes.target[-30:]   #Features and label should be same

model=linear_model.LinearRegression()

model.fit(diabetes_X_train,diabetes_y_train)                     #fit our data

diabetes_y_predict=model.predict(diabetes_X_test)

# Mean sq error, weights,intercept:

print("Mean Squared Error:",mean_squared_error(diabetes_y_test,diabetes_y_predict))
print("Weights:",model.coef_)
print("Intercept:",model.intercept_)

'''plt.scatter (diabetes_X_test,diabetes_y_test)           #can be used to show best fit 
plt.plot(diabetes_X_test,diabetes_y_predict)
plt.show()'''







#print(diabetes.keys())
#(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
#print (diabetes.DESCR)