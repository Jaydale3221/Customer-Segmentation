#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd

df=pd.read_csv("/Users/jaydale/Desktop/code/Customer Segmentation/data.csv",encoding='latin1')

print(df.head(10))


# In[18]:


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

# # Read the dataset
# df = pd.read_csv('data.csv')

# Clean the data
# Drop NA values
df = df.dropna()

# Remove unnecessary columns
df = df.drop(columns=['InvoiceNo', 'Description', 'CustomerID', 'Country'])

# Convert Quantity and UnitPrice to numeric
df['Quantity'] = pd.to_numeric(df['Quantity'])
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'])

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])


# In[19]:


# # Perform basic EDA
# Histogram of Quantity
plt.hist(df['Quantity'], bins = 10)
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.title('Histogram of Quantity')
plt.show()

# Histogram of UnitPrice
plt.hist(df['UnitPrice'], bins = 10)
plt.xlabel('UnitPrice')
plt.ylabel('Frequency')
plt.title('Histogram of UnitPrice')
plt.show()


# In[22]:


#replace empty values in 'Description' column with 'N/A'
df.fillna({'Description':'N/A'}, inplace=True)

#convert 'UnitPrice' and 'Quantity' columns to float type
df['UnitPrice'] = df['UnitPrice'].astype(float)
df['Quantity'] = df['Quantity'].astype(float)

#drop duplicates
df.drop_duplicates(inplace=True)

#print pre-processed data
print(df)


# In[ ]:




