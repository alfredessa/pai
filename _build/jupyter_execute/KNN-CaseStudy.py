#!/usr/bin/env python
# coding: utf-8

# # K Nearest Neighbors

# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# ## Load Data

# In[2]:


df = pd.read_csv("data/loans.csv").round(1)


# In[3]:


df.head()


# ## Run Classifier

# In[4]:


# extract X,y from pandas
X = df.drop(['Status'],axis=1)
y = df['Status']


# In[5]:


# set classifiers for k =3,5
knn3 = KNeighborsClassifier(n_neighbors=3).fit(X,y)
knn5 = KNeighborsClassifier(n_neighbors=5).fit(X,y)


# ## Evaluate Classifier

# In[6]:


y_pred = knn3.predict(X)
print(accuracy_score(y,y_pred))


# In[7]:


print(confusion_matrix(y,y_pred))


# In[8]:


print(classification_report(y,y_pred))


# ## Predictions

# In[9]:


# define new points for which we want prediction
new_points =[[82000,530],[123000,510],[90000,670],[99000,610]]


# In[10]:


y_pred_3 = knn3.predict(new_points)
y_pred_5 = knn5.predict(new_points)


# In[11]:


# print predictions
print(y_pred_3)
print(y_pred_5)


# In[12]:


data = {'KNN=3': y_pred_3, 'KNN=5': y_pred_5}
pd.DataFrame.from_dict(data, orient='index',
                       columns=['Point1','Point2','Point3','Point4'])


# In[ ]:




