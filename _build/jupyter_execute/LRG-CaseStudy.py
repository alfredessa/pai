#!/usr/bin/env python
# coding: utf-8

# # Practical Artificial Intelligence
# ## Logistic Regression
# 

# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# ## Read and Confirm Data

# In[2]:


df = pd.read_csv('data/ccdefault.csv').round(1)


# In[3]:


df.head()


# In[4]:


# remap default = 'Yes' to 1; 'No' to 0
df['default'] = np.where(df['default'] == "Yes", 1, 0)
df['student'] = np.where(df['student'] == "Yes", 1, 0)


# In[5]:


# size balance and income to be 100s of $
df['balance'] = np.round(df['balance']/100,0)
df['income'] = np.round(df['income']/100,0)


# In[6]:


df.head()


# ## Logistic Regression 

# In[7]:


lr = smf.logit(formula='default ~ balance + C(student)',data=df).fit() 


# In[8]:


lr.summary()


# ## Evaluate Model

# In[9]:


X = df[['balance','student']]
y = df['default']
y_probabilities = lr.predict(X)


# In[10]:


y_hat = list(map(round,y_probabilities))


# In[11]:


print(accuracy_score(y,y_hat))


# In[12]:


print(confusion_matrix(y,y_hat))


# In[13]:


print(classification_report(y,y_hat))


# ## Predictions

# In[14]:


# predict new points
data_new = {'balance': [5.2,10.1,12.3,20.1,22.6], 
        'student': [1,0,1,0,1]}
df_new = pd.DataFrame(data_new)


# In[15]:


df_new['probability'] = lr.predict(df_new).round(2)


# In[16]:


df_new


# In[17]:


np.exp(lr.params) 


# In[18]:


(np.exp(lr.params)-1)*100

