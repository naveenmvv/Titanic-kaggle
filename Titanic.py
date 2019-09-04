# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 20:02:33 2019

@author: NAVEEN
"""

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('E:/Data/titanic/train.csv')
label = LabelEncoder()

dr = df.iloc[:,[2,4,5,6,7,9]]
dt = df.iloc[:,1]
#dr['Age'].fillna((dr['Age'].mean()), inplace=True)
dr['Sex'] = dr['Sex'].map({'female': 1, 'male': 0})
#dr['Embarked'] = dr['Embarked'].map({'C': 0, 'S': 1, 'Q':2})
dr['Age'].fillna(dr['Age'].median(), inplace = True)
dr['Age'] = pd.cut(dr['Age'].astype(int), 5)
dr['Age'] = label.fit_transform(dr['Age'])

dr['FamilySize'] = dr ['SibSp'] + dr['Parch'] + 1
dr['IsAlone'] = 1 #initialize to yes/1 is alone
dr['IsAlone'].loc[dr['FamilySize'] > 1] = 0
dr = dr.drop(['SibSp','Parch','FamilySize'],axis=1)    
corr = dr.corr()
X = dr
Y = dt

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
model = XGBClassifier(loss='exponential', learning_rate=0.1,n_estimators=110,max_depth=4)
#model = LogisticRegression(random_state=2)
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

dtest = pd.read_csv('E:/Data/titanic/test.csv')
dtest = dtest[['Pclass','Sex','Age','SibSp','Parch','Fare']]
dtest['Sex'] = dtest['Sex'].map({'female': 1, 'male': 0})
#dtest['Embarked'] = dtest['Embarked'].map({'C': 0, 'S': 1, 'Q':2})
dtest['Age'].fillna(dtest['Age'].median(), inplace = True)
dtest['Age'] = pd.cut(dtest['Age'].astype(int), 5)
dtest['Age'] = label.fit_transform(dtest['Age'])

dtest['FamilySize'] = dtest ['SibSp'] + dtest['Parch'] + 1
dtest['IsAlone'] = 1 #initialize to yes/1 is alone
dtest['IsAlone'].loc[dtest['FamilySize'] > 1] = 0
dtest = dtest.drop(['SibSp','Parch','FamilySize'],axis=1) 

y_prod = list(model.predict(dtest))