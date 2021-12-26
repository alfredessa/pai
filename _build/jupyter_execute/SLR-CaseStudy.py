#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression

# ## Import Libraries

# In[1]:


# import libraries
import pandas as pd
from statsmodels.formula.api import ols


# ## Load and Verify Dataset

# In[2]:


# load dataset and create dataframe
df = pd.read_csv('data/edincome.csv').round(1)


# In[3]:


# verify first few records
df.head()


# ## Run Regression

# In[4]:


slr = ols('Income ~ Education',df).fit()


# ## Review Results and Evaluate Model

# In[5]:


print(slr.params)


# In[6]:


slr.summary()


# In[7]:


print(slr.rsquared)


# In[8]:


print(slr.mse_model)


# ## Generate Predictions

# In[9]:


# predict new points
data = {'Education': [12,16,18]}
df_predict = pd.DataFrame(data).round(1)


# In[10]:


df_predict['Income'] = slr.predict(df_predict).round(1)


# In[11]:


df_predict

