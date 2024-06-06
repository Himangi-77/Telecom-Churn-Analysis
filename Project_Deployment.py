#!/usr/bin/env python
# coding: utf-8

# In[14]:


import streamlit as st
import pandas as pd
import numpy as np


# In[18]:


st.title('TELECOM CHURN ANALYSIS')
st.sidebar.header('User Input Parameters')


# In[34]:


def user_input_features():
    area_code = st.sidebar.selectbox("area code",('408','415','510'))    
    voice_plan = st.sidebar.selectbox('Voice Plan',('1','0'))
    voice_messages = st.sidebar.number_input("Number of Voice Messages")
    intl_plan = st.sidebar.selectbox('International Plan',('1','0'))
    intl_mins = st.sidebar.number_input("Minutes of International Call")
    intl_calls = st.sidebar.number_input("Number of International Calls")
    day_mins = st.sidebar.number_input("Minutes of Day Call")
    eve_calls = st.sidebar.number_input("Number of Evening Calls")
    eve_charge = st.sidebar.number_input("Charge of Evening Call")
    night_calls = st.sidebar.number_input("Number of Night Calls")
    customer_calls = st.sidebar.number_input("Number of Customer Calls")
    data = {'Area Code':area_code,
           'Voice Plan':voice_plan,
           'Voice Messages':voice_messages,
           'International Plan':intl_plan,
           'International Minutes':intl_mins,
           'International Calls':intl_calls,
           'Day Minutes':day_mins,
           'Evening Calls':eve_calls,
           'Evening Charge':eve_charge,
           'Night Calls':night_calls,
           'Customer Calls':customer_calls}
    features = pd.DataFrame(data,index = [0])
    return features 


# In[35]:


df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


# In[36]:


churn = pd.read_csv("C:/Users/HIMANGI/Downloads/export_churn.csv")
df["area code"] = df["area code"].astype('int')
df["voice mail plan"] = df["voice mail plan"] .astype('int')
df["international plan"] = df["international plan"] .astype('int')


# In[37]:


x = churn.iloc[:,0:11]
y = churn.iloc[:,-1]


# In[38]:

from xgboost import XGBClassifier
xgb_model=XGBClassifier().fit(x,y)


# In[40]:


prediction = xgb_model.predict(df)
prediction_proba = xgb_model.predict_proba(df)


# In[42]:


st.subheader('Will the customer churn?')
st.write('Yes, The Customer will CHURN' if prediction_proba[0][1] > 0.5 else 'No, The Customer will NOT CHURN')

# In[43]:


st.subheader('Prediction Probability')
st.write(prediction_proba)


# In[ ]:




