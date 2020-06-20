
# coding: utf-8

# In[1]:


import graphlab


# In[3]:


data=graphlab.SFrame("prices.csv")


# In[4]:


data


# In[5]:


train_data,test_data=data.random_split(.8,seed=0)


# In[6]:


stock_model=graphlab.linear_regression.create(train_data,target='close',features=['date','open','low','high'])


# In[18]:





# In[19]:


import matplotlib.pyplot as mt
get_ipython().magic(u'matplotlib inline')


# In[25]:


mt.plot(test_data['high'],test_data['close'],'.',test_data['high'],stock_model.predict(test_data),'.')


# In[ ]:




