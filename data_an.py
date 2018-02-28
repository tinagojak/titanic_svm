#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 01:57:29 2018

@author: tina
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix

def uredi_podatke(data_train):
    
    #analiza ucitanih podataka
    #print(data_train.columns.values)
    #print(data_train.describe(include='all'))
    #print(data_test.info())
    #print(data_train.isnull().sum())
    print(data_train[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean().sort_values(by='Survived', ascending=False))
    
    #uredivanje podataka
    #x_train, y_train = data_train.drop(['Survived', 'PassengerId', 'Name'], axis=1), data_train['Survived']
    x_train, y_train = data_train.drop(['Survived', 'PassengerId', 'Ticket', 'Cabin', 'Name'], axis=1), data_train['Survived']
    x_train['Sex'] = x_train['Sex'].map({"male": 0, "female": 1})
    x_train['Age'].fillna(x_train.Age.median(), inplace=True)
    x_train['Embarked'].fillna('S', inplace=True)       #'S' je najcesci
    x_train['Family'] = x_train['Parch'] + x_train['SibSp']
    x_train['IsAlone'] = 0
    x_train.loc[x_train['Family'] == 0, 'IsAlone'] = 1
    x_train = x_train.drop(['Family', 'Parch', 'SibSp'], axis=1)
    x_train['Embarked'] = x_train['Embarked'].map({"S": 0, "C": 1, "Q":2}).astype(int)
    guess_ages = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = x_train[(x_train['Sex'] == i) & \
                               (x_train['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
    
            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5        
    for i in range(0, 2):
        for j in range(0, 3):
            x_train.loc[ (x_train.Age.isnull()) & (x_train.Sex == i) & (x_train.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]
    x_train['Age'] = x_train['Age'].astype(int)
    x_train['AgeBand'] = pd.cut(x_train['Age'], 5)
    x_train.loc[ x_train['Age'] <= 16, 'Age'] = 0
    x_train.loc[(x_train['Age'] > 16) & (x_train['Age'] <= 32), 'Age'] = 1
    x_train.loc[(x_train['Age'] > 32) & (x_train['Age'] <= 48), 'Age'] = 2
    x_train.loc[(x_train['Age'] > 48) & (x_train['Age'] <= 64), 'Age'] = 3
    x_train.loc[ x_train['Age'] > 64, 'Age']
    x_train = x_train.drop(['AgeBand'], axis=1)
    
    x_train['FareBand'] = pd.qcut(x_train['Fare'], 4)
    x_train.loc[ x_train['Fare'] <= 7.91, 'Fare'] = 0
    x_train.loc[(x_train['Fare'] > 7.91) & (x_train['Fare'] <= 14.454), 'Fare'] = 1
    x_train.loc[(x_train['Fare'] > 14.454) & (x_train['Fare'] <= 31), 'Fare']   = 2
    x_train.loc[ x_train['Fare'] > 31, 'Fare'] = 3
    x_train['Fare'] = x_train['Fare'].astype(int)
    
    x_train = x_train.drop(['FareBand'], axis=1)
    
    #Podjela podataka za train i val
    msk = np.random.rand(len(x_train)) < 0.8
    x_train_ = x_train[msk]
    x_val = x_train[~msk]
    
    y_train_ = y_train[msk]
    y_val = y_train[~msk]

    
#    x_test = data_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
#    x_test['Sex'] = x_test['Sex'].map({"male": 0, "female": 1})
#    x_test['Age'].fillna(x_test.Age.median(), inplace=True)
#    x_test['Fare'].fillna(x_test.Fare.median(), inplace=True)
#    x_test['Family'] = x_test['Parch'] + x_test['SibSp']
#    x_test['IsAlone'] = 0
#    x_test.loc[x_test['Family'] == 0, 'IsAlone'] = 1
#    x_test = x_test.drop(['Family', 'Parch', 'SibSp'], axis=1)
#    x_test['Embarked'] = x_test['Embarked'].map({"S": 0, "C": 1, "Q":2}).astype(int)
#    x_test['Fare'].fillna(x_test['Fare'].dropna().median(), inplace=True)
    
    return x_train_, y_train_, x_val, y_val