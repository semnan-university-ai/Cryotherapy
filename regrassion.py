#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/Cryotherapy
# dataset link : http://archive.ics.uci.edu/ml/datasets/Cryotherapy+Dataset+
# email : amirsh.nll@gmail.com


# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression


# In[2]:


col_names= ['Result_of_Treatment','sex', 'age', 'Time', 'Number_of_Warts', 'Type', 'Area' ]
cry=pd.read_csv("Cryotherapy.csv",header=None, names=col_names)


# In[3]:


inputs =cry.drop('Result_of_Treatment',axis='columns')
target =cry['Result_of_Treatment']


# In[4]:


input_train,input_test,target_train,target_test=train_test_split(inputs,target,test_size=0.3,random_state=1)


# In[5]:


reg = LogisticRegression()
reg.fit(input_train,target_train)
y_pred=reg.predict(input_test)


# In[6]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result1 = classification_report(target_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(target_test,y_pred)
print("Accuracy:",result2)

