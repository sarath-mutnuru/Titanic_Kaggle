# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 14:38:36 2019

@author: SARATHKUMAR
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV
#%%
# loading data #
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
#%% data cleaning
# Removing Name,Cabin,Ticket as they have no effect on Survival
train.drop(['Name','Cabin','Ticket','PassengerId'],axis=1,inplace=True)

test.drop(['Name','Cabin','Ticket'],axis=1,inplace=True)

# Dealing with missing entries - filling mode
cols_train=train.columns[pd.isnull(train).any()].tolist()
for c in cols_train:
    
    val=train[c].mode().tolist()[0]
    train[c].fillna(val,inplace=True)

    
cols_test=test.columns[pd.isnull(test).any()].tolist()
for c in cols_test:
    val=test[c].mode().tolist()[0]
    test[c].fillna(val,inplace=True)
    
# extracting X_train, y_train, X_test from dfs
import numpy as np
y=np.asarray(train['Survived'].tolist())
train.drop(['Survived'],axis=1,inplace=True)

# changing categorical variables to numbers
from sklearn.preprocessing import LabelEncoder
emb_LE=LabelEncoder()
train['Embarked_labels']=emb_LE.fit_transform(train['Embarked'])
test['Embarked_Labels']=emb_LE.transform(test['Embarked'])
sex_LE=LabelEncoder()
train['Sex_labels']=emb_LE.fit_transform(train['Sex'])
test['Sex_Labels']=emb_LE.transform(test['Sex'])
train.drop(['Embarked','Sex'],axis=1,inplace=True)
test.drop(['Embarked','Sex'],axis=1,inplace=True)

# clubbing a certain range of ages and Fares
train['nAge']=train['Age'].apply(lambda x:0 if x<=16 else(1 if x<=26 else (2 if x<=36 else(3 if x<=62 else 4))))
train.drop(['Age'],axis=1,inplace=True)
train['nFare']=train['Fare'].apply(lambda x:0 if x<=17 else(1 if x<=30 else (2 if x<=100 else 3)))
train.drop(['Fare'],axis=1,inplace=True)
test['nAge']=test['Age'].apply(lambda x:0 if x<=16 else(1 if x<=26 else (2 if x<=36 else(3 if x<=62 else 4))))
test.drop(['Age'],axis=1,inplace=True)
test['nFare']=test['Fare'].apply(lambda x:0 if x<=17 else(1 if x<=30 else (2 if x<=100 else 3)))
test.drop(['Fare'],axis=1,inplace=True)

train['Family-Size']=train['SibSp']+train['Parch']+1
test['Family-Size']=test['SibSp']+test['Parch']+1
test.drop(['SibSp','Parch'],axis=1,inplace=True)
train.drop(['SibSp','Parch'],axis=1,inplace=True)

X=train.values
test_passId=test['PassengerId']
test.drop(['PassengerId'],axis=1,inplace=True)
X_test=test.values
#%% Training
# Validation set split
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.25, random_state = 10)

# Scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
#X_train_scaled=X_train
X_train_scaled=scaler.fit_transform(X_train)
# Training the model
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# =============================================================================
# parameters = {
#     "loss":["deviance"],
#     "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
#     "min_samples_split": np.linspace(0.1, 0.5, 12),
#     "min_samples_leaf": np.linspace(0.1, 0.5, 12),
#     "max_depth":[3,5,8],
#     "max_features":["log2","sqrt"],
#     "criterion": ["friedman_mse",  "mae"],
#     "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
#     "n_estimators":[10]
#     }
# =============================================================================
# =============================================================================
# parameters = {
#     "loss":["deviance"],
#     "learning_rate": [0.5],
#     "min_samples_split": np.linspace(0.1, 0.5, 12),
#     "min_samples_leaf": np.linspace(0.1, 0.5, 12),
#     "max_depth":[5],
#     "max_features":["log2","sqrt"],
#     "criterion": ["friedman_mse",  "mae"],
#     "subsample":[1.0],
#     "n_estimators":[10],
#     "random_state":[10]
#     }
# =============================================================================
parameters = {
    "alpha":[0.01,0.001],
    "learning_rate_init": [0.01,0.1],
    "hidden_layer_sizes":[(10,2),(10,2,3),(10,5)],
    "solver":['sgd','adam'],
    "momentum":[0.1,0.5,0.9],
    "learning_rate":["constant","adaptive"],
    "max_iter":[1000]
    }
#classifier=linear_model.LogisticRegression(C=1,max_iter=10,verbose=10)
#classifier=SVC(C=100,gamma=0.01,kernel='rbf')
#classifier=DecisionTreeClassifier(random_state=10)
#classifier = GridSearchCV(MLPClassifier(), parameters, cv=10, n_jobs=-1)
classifier=MLPClassifier(hidden_layer_sizes=(50,50),alpha=1,max_iter=1000,solver='sgd',momentum=0.9,verbose=10,learning_rate_init=0.005,learning_rate='adaptive')
#classifier=AdaBoostClassifier(n_estimators=100,random_state=40,learning_rate=0.9)
#classifier=GradientBoostingClassifier(subsample=1,max_depth=5,n_estimators=10,random_state=20,learning_rate=0.5)

#classifier = RandomForestClassifier(n_estimators = 150, criterion = 'entropy', random_state = 40)
#classifier = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)


classifier.fit(X_train_scaled, y_train)
print('on training data',classifier.score(X_train_scaled, y_train))
#print(classifier.best_params_)

# Testing on Validation Set
X_val_scaled=X_val
X_val_scaled=scaler.transform(X_val)
y_val_pred = classifier.predict(X_val_scaled)
# plotting Confusion matrix
import pandas as pd
from  sklearn.metrics import f1_score
print(pd.crosstab(y_val, y_val_pred, rownames=['Actual labels'], colnames=['Predicted Lables']))
print ('F1score',f1_score(y_val,y_val_pred))
#%% Testing
#X_test_scaled=X_test
X_test_scaled=scaler.transform(X_test)
y_pred=classifier.predict(X_test_scaled)
result=pd.DataFrame({'PassengerId':test_passId,'Survived':y_pred},columns=['PassengerId','Survived'])
result.to_csv("result.csv",index=False,sep=",")