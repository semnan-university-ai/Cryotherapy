#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/Cryotherapy
# dataset link : http://archive.ics.uci.edu/ml/datasets/Cryotherapy+Dataset+
# email : amirsh.nll@gmail.com


# In[64]:


import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split


# In[65]:


col_names= ['Result_of_Treatment','sex', 'age', 'Time', 'Number_of_Warts', 'Type', 'Area' ]
cry= pd.read_csv("Cryotherapy.csv",header=None, names=col_names)


# In[66]:


inputs =cry.drop('sex',axis='columns')
target =cry['Result_of_Treatment']


# In[67]:


input_train, input_test, target_train, target_test = train_test_split(inputs, target, test_size=0.3, random_state=1)


# In[95]:


from sklearn.neural_network import MLPClassifier 
mlp = MLPClassifier(hidden_layer_sizes=(7,6), max_iter=5000)
mlp.fit(input_train, target_train)


# In[96]:


from sklearn.metrics import accuracy_score
predictions_train =mlp.predict(input_train)
print("accuracy for train data: ", accuracy_score(predictions_train, target_train))
y_pred=mlp.predict(input_test)
print("accuracy for test data: ", accuracy_score(y_pred, target_test))


# In[97]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result1 = classification_report(target_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(target_test,y_pred)
print("Accuracy:",result2)

