#!/usr/bin/env python
# coding: utf-8

# 
# # Multiple Linear Regression
# 

# ## Load Libraries

# In[1]:



import pandas as pd
from statsmodels.formula.api import ols


# ## Load and Verify Data

# In[2]:


df = pd.read_csv("data/academicperformance.csv")
df.head()


# In[ ]:





# ## Multiple Linear Regression

# In[3]:


mlr = ols('Grade ~ GPA + Sleep + Time', df).fit()


# In[4]:


mlr.summary()


# ## Predictions

# In[5]:


data = {'GPA':[3,3,3,2,3,4,2.5,2.5,2.5],
        'Sleep':[5,6,7,6,6,6,5,5,5],
        'Time':[30,30,30,30,30,30,40,50,60]}
df_predict = pd.DataFrame(data)


# In[6]:


df_predict['Grade'] = mlr.predict(df_predict).round(1)


# In[7]:


df_predict

