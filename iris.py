# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:24:59 2024

@author: koralsenturk
"""

# %%

#Load Dataset
from  sklearn import  datasets
dataSet=datasets.load_iris()
features = dataSet.data
labels = dataSet.target
labelsNames =list(dataSet.target_names)
featureNames = dataSet.feature_names
#print(featureNames)

#print([labelsNames[i] for i in labels[-3:]]) #son 3 label getirilecek

# %%
#Analyze Data

import pandas as pd
featuresDF = pd.DataFrame(features)
featuresDF.columns = featureNames

#print(type(featuresDF))
#print(featuresDF.describe())

# %%
#Visualize Data

#featuresDF.plot(kind = "box")


# %%

#Select Model
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=8)


# %%
#Split Dataset

import numpy as np
from sklearn.model_selection import train_test_split
X = features
y = labels

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.33, random_state=42)

#print(X_test[:3])
# %%
#Train Model

clf.fit(X_train, y_train)

# %%
#Test Model

accuracy = clf.score(X_test, y_test)
print("Accuracy on test data: ", accuracy)

# %%
#Improve Model
#Algoritmanın ayarlarını değiştirmek, neighbor sayısının değiştirilmesi gibi.

# %%
#Model Save
from joblib import dump, load
filename = "myFirstSavedModel.joblib"
dump(clf, filename)

# %%
#Load Model

clf = load(filename)
# %%


