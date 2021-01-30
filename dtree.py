#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/Cryotherapy
# dataset link : http://archive.ics.uci.edu/ml/datasets/Cryotherapy+Dataset+
# email : amirsh.nll@gmail.com


# In[3]:


import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics


# In[4]:


columns =['Result_of_Treatment','sex', 'age', 'Time', 'Number_of_Warts','Type','Area']
cry= pandas.read_csv("Cryotherapy.csv",header=None, names=columns)


# In[5]:


print(cry)


# In[6]:


inputs =cry.drop('Result_of_Treatment',axis='columns')
target =cry['Result_of_Treatment']


# In[7]:


print(inputs)


# In[8]:


input_train,input_test,target_train,target_test=train_test_split(inputs,target,test_size=0.3,random_state=1)


# In[9]:


print (input_train.shape, target_train.shape)
print (input_test.shape, target_test.shape)


# In[10]:


dtree = DecisionTreeClassifier()
dtree = dtree.fit(input_train,target_train)
y_pred =dtree.predict(input_test)
y_pred


# In[12]:


from sklearn.metrics import classification_report, accuracy_score
result1 = classification_report(target_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(target_test,y_pred)
print("Accuracy:",result2)


# In[25]:


from sklearn import tree
tree.plot_tree(dtree.fit(inputs,target))

