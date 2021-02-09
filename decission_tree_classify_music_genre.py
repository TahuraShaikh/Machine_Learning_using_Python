
#Simple decission tree algorithm on a dataset consisting of 20 datapoints

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data=pd.read_csv("music.csv")   # music csv file created with just 20 records to check working of classifer
X=music_data.drop(columns="genre") #genre column used as label and first 2 columns used as features
y=music_data["genre"]
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)


model=DecisionTreeClassifier()
model.fit(X_train,y_train)
predictions=model.predict(X_test)

score=accuracy_score(y_test, predictions)
score

#joblib dump :

sic_data=pd.read_csv("music.csv")
X=music_data.drop(columns="genre")
y=music_data["genre"]

model=DecisionTreeClassifier()
model.fit(X,y)

joblib.dump(model,"music_recommender.joblib")


#joblib load:

model=joblib.load("music_recommender.joblib")
predictions=model.predict([[22,1]])
predictions