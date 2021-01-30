#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/Cryotherapy
# dataset link : http://archive.ics.uci.edu/ml/datasets/Cryotherapy+Dataset+
# email : amirsh.nll@gmail.com


# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[22]:


col_names= ['Result_of_Treatment','sex', 'age', 'Time', 'Number_of_Warts', 'Type', 'Area' ]
cry=pd.read_csv("Cryotherapy.csv",header=None, names=col_names)


# In[23]:


inputs =cry.drop('Result_of_Treatment',axis='columns')
target =cry['Result_of_Treatment']


# In[24]:


input_train,input_test,target_train,target_test=train_test_split(inputs,target,test_size=0.3,random_state=1)


# In[25]:


from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
scaler.fit(input_train)
input_train=scaler.transform(input_train)
input_test =scaler.transform(input_test)


# In[51]:


from sklearn.neighbors import KNeighborsClassifier
best=[]
k=[1, 3, 5, 7, 9]
for i in range(len(k)):  
    classifier = KNeighborsClassifier(n_neighbors=k[i])
    classifier.fit(input_train,target_train)
    y_pred = classifier.predict(input_test)
    y_pred
    result1 = classification_report(target_test,y_pred)
    print("Classification Report:",)
    print (result1)
    result2 = accuracy_score(target_test,y_pred)
    print("Accuracy",k[i],":",result2)

