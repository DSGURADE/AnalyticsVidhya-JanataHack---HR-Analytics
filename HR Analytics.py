#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# ### Data Reading

# In[2]:


# Read the csv file using 'read_csv'
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[3]:


# Top 5 records of train dataframe
train.head()


# In[4]:


# Check the number of rows and columns in the train dataframe
train.shape


# In[5]:


# Check the column-wise info of the train dataframe
train.info()


# ### Data Visualization

# In[6]:


sns.countplot(train['target'])
plt.show()


# In[7]:


# See the target frequency with the Insurence type
pd.crosstab(train['experience'],train['target'] ).plot(kind="bar",figsize=(15,6))
plt.title('Frequency of target variable with experience')
plt.xticks(rotation=0)
plt.xlabel('Experience')
plt.ylabel('Frequency')
plt.show()


# In[8]:


# Outlier analysis for debt_to_ratio with target variable
sns.boxplot(x='target',y = 'city_development_index',data=train)


# In[9]:


sns.countplot(train['gender'])
plt.show()


# In[10]:


# See the target frequency with the Insurence type
pd.crosstab(train['company_type'],train['target'] ).plot(kind="bar",figsize=(15,6))
plt.title('Frequency of target variable with company_type')
plt.xticks(rotation=0)
plt.xlabel('Company Type')
plt.ylabel('Frequency')
plt.show()


# In[11]:


# See the target frequency with the Insurence type
pd.crosstab(train['education_level'],train['target'] ).plot(kind="bar",figsize=(15,6))
plt.title('Frequency of target variable with Education Level')
plt.xticks(rotation=0)
plt.xlabel('Education Level')
plt.ylabel('Frequency')
plt.show()


# ### Data cleaning and Data Prepration

# In[12]:


train.isnull().sum()


# In[13]:


round(100*(train.isnull().sum()/len(train)),2)


# In[14]:


test.isnull().sum()


# In[15]:


round(100*(test.isnull().sum()/len(test)),2)


# In[16]:


train['major_discipline'].value_counts()


# In[17]:


train['company_size'].value_counts()


# In[18]:


train['gender'].fillna("Male", inplace=True)
train['major_discipline'].fillna("STEM", inplace=True)
train['company_type'].fillna("Pvt Ltd", inplace=True)
train['company_size'].fillna("50-990", inplace=True)
test['gender'].fillna("Male", inplace=True)
test['major_discipline'].fillna("STEM", inplace=True)
test['company_type'].fillna("Pvt Ltd", inplace=True)
test['company_size'].fillna("50-990", inplace=True)


# In[19]:


# Dropping the null values from the dataframe
train.dropna(inplace=True)


# In[20]:


train.isnull().sum()


# In[21]:


train.drop(['enrollee_id','city'], axis=1, inplace=True)
test.drop(['enrollee_id','city'], axis=1, inplace=True)


# In[22]:


# Label Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

for x in train.columns:
    if train[x].dtype == type(object):
        train[x] = train[x].fillna('NaN')
        test[x] = test[x].fillna('NaN')
        encoder = LabelEncoder()
        encoder.fit(list(set(list(train[x]) + list(test[x]))))
        train[x] = encoder.transform(train[x])
        test[x] = encoder.transform(test[x])


# In[23]:


train.head()


# In[24]:


test.head()


# In[25]:


# plot training dataset heatmap
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[26]:


X = train.drop(['target'],axis=1)
y = train['target']


# In[27]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
test_data = sc.transform(test.copy().values)


# In[28]:


# XGB Classifier
from xgboost import XGBClassifier

xgb = XGBClassifier( learning_rate =0.1,
 n_estimators=112,
 max_depth=9,
 min_child_weight=5,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.6,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=13,
 reg_lambda=5,
# max_delta_step=1,
 alpha=0,
 base_score=0.5,
 seed=1029)

xgb.fit(X_train, y_train)

# Predicting the Test set results
y_pred = xgb.predict(X_test)  

# Accuracy of XGB model
accuracy_xgb = round(xgb.score(X_train, y_train) * 100, 2)
print("Accuracy score of XGB algorithm is:", accuracy_xgb)


# In[29]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[30]:


# Predicting the Test set results
test_pred = xgb.predict(test_data)


# In[31]:


# load session_id of test dataset
test_enrollee_id = pd.read_csv('test.csv')['enrollee_id']
print(test_enrollee_id.shape)


# In[32]:


# save results to csv
submission_file = pd.DataFrame({'enrollee_id': test_enrollee_id, 'target': test_pred})
submission_file = submission_file[['enrollee_id','target']] 
submission_file.to_csv('Final_Solution.csv', index=False)


# In[ ]:




