#!/usr/bin/env python
# coding: utf-8

# In[179]:


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose



df=pd.read_csv("/Users/jaydale/Desktop/code/Customer Segmentation/data.csv",encoding='latin1')

print(df.head(10))


# In[180]:


df['CustomerID'] = df['CustomerID'].astype('category')
df['CustomerID'] = df['CustomerID'].cat.codes


# In[181]:


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


# In[182]:


# # Perform basic EDA
# Histogram of Quantity
plt.hist(df['Quantity'], bins = 10)
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.title('Histogram of Quantity')
plt.show()


# In[183]:


# Histogram of UnitPrice
plt.hist(df['UnitPrice'], bins = 10)
plt.xlabel('UnitPrice')
plt.ylabel('Frequency')
plt.title('Histogram of UnitPrice')
plt.show()


# In[184]:


#replace empty values in 'Description' column with 'N/A'
df.fillna({'Description':'N/A'}, inplace=True)

#convert 'UnitPrice' and 'Quantity' columns to float type
df['UnitPrice'] = df['UnitPrice'].astype(float)
df['Quantity'] = df['Quantity'].astype(float)

#drop duplicates
df.drop_duplicates(inplace=True)

#print pre-processed data
print(df)


# In[185]:


# create a box plot of the Quantity column
fig, ax = plt.subplots(figsize=(5,5))
ax.boxplot(df['Quantity'], showfliers=False, patch_artist=True) # remove outliers

# add x-axis label and title
ax.set_xlabel("Quantity")
ax.set_title("Box Plot of Quantity Column")

# change color and line width of the box plot
colors = ['blue']
for patch, color in zip(ax.artists, colors):
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .3))

# add grid to the plot
ax.grid(linestyle='--')

# show the plot
plt.show()


# In[186]:


# Calculate the mean of the numerical variables
print("Mean:")
print(df[['Quantity', 'UnitPrice']].mean())


# In[187]:


# # Calculate the median of the numerical variables
print("\nMedian:")
print(df[['Quantity', 'UnitPrice']].median())


# In[188]:


# # Calculate the mode of the numerical variables
print("\nMode:")
print(df[['Quantity', 'UnitPrice']].mode())


# In[189]:


# # Calculate the standard deviation of the numerical variables
print("\nStandard Deviation:")
print(df[['Quantity', 'UnitPrice']].std())


# In[190]:


# # Calculate the frequency of the categorical variables
print("\nFrequency:")
for col in [ 'StockCode', 'InvoiceDate']:
    print(f"\n{col}:")
    print(df[col].value_counts())


# In[191]:


#  Convert the InvoiceDate column into a datetime object
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Set the InvoiceDate column as the index of the dataframe
df.set_index('InvoiceDate', inplace=True)

# Aggregate the data by summing up the quantity for each day
df_daily = df.resample('D').sum()

# Plot the time series data
plt.figure(figsize=(12, 8))
plt.plot(df_daily['Quantity'])
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.title('Time Series of Daily Quantity')
plt.grid(True)
plt.show()


# Decompose the time series data into its trend, seasonal, and residual components
 
result = seasonal_decompose(df_daily['Quantity'], model='additive')
result.plot()
plt.show()


# In[192]:


# # Remove Outliners
# Calculate the z-scores for each numerical variable
z = np.abs(stats.zscore(df[['Quantity', 'UnitPrice']]))

# Keep only the rows with z-score less than 3 (3 is a common threshold for outliers)
df = df[(z < 3).all(axis=1)]


# In[193]:


# Select the relevant columns for clustering
X = df[['Quantity', 'UnitPrice']]

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
df['Cluster'] = kmeans.predict(X)

# Plot the scatter plot for Quantity and UnitPrice with different colors for each cluster
plt.scatter(df['Quantity'], df['UnitPrice'], c=df['Cluster'], cmap='rainbow', s=50, alpha=0.7)
plt.xlabel('Quantity')
plt.ylabel('UnitPrice')
plt.title('K-means Clustering Results')
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




