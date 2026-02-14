#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats

# Generate synthetic dataset for testing
np.random.seed(42)
n_samples = 100
data = {
    'income': np.random.normal(50000, 15000, n_samples),
    'credit_score': np.random.normal(650, 50, n_samples),
    'job_title': np.random.choice(['Engineer', 'Teacher', 'Doctor', 'Artist'], n_samples),
    'target': np.random.choice([0, 1], n_samples)
}
# Introduce missing values and outliers for testing
data['income'][np.random.randint(0, n_samples, 5)] = np.nan
data['credit_score'][np.random.randint(0, n_samples, 3)] = np.nan
data['income'][np.random.randint(0, n_samples, 2)] = 150000  # Outliers
df = pd.DataFrame(data)


# In[2]:


# Handle missing values by filling with median
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    df[column].fillna(df[column].median(), inplace=True)


# In[3]:


# Remove duplicates
df.drop_duplicates(inplace=True)


# In[4]:


scaler = StandardScaler()
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_features] = scaler.fit_transform(df[numeric_features])


# In[5]:


# One-hot encode categorical features
df = pd.get_dummies(df, drop_first=True)


# In[6]:


# Detect and remove outliers using Z-score
#Keep only rows where all Z-scores are less than 3
z_scores = np.abs(stats.zscore(df.select_dtypes(include=['float64', 'int64'])))
df = df[(z_scores < 3).all(axis=1)]


# In[7]:


# Apply log transformation to skewed data
df['income_log'] = np.log1p(df['income'])


# If someone earns:
# 
# 50,000 → becomes log(50,001)
# 
# 150,000 → becomes log(150,001)

# In[8]:


#Separate the data into training and testing sets:
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# This guide walks through essential data preprocessing steps used in real-world machine learning projects. You learned how to clean the dataset by fixing missing values, removing duplicates, and correcting errors. Numerical features such as income and credit score are standardized, while categorical features like job titles are converted into numeric form. Outliers are detected and removed to improve model stability, and skewed data is transformed using log techniques for better distribution. Finally, the cleaned and enhanced dataset is split into training and testing sets to ensure fair model evaluation

# 
