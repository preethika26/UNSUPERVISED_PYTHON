# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 18:21:41 2019

@author: Preethika
"""

import pandas as pd
credit=pd.read_csv('D:\\ssn\\unsupervised r code\\CLUSTERING-CREDIT CARD DATASET\\CC GENERAL.csv')

##remove first column
credit.drop(['CUST_ID'], axis = 1,inplace=True) 

##check any missing value
credit.isna().sum()

##find mean and median
credit.mean()
credit.median()

##fill the missing values
credit['MINIMUM_PAYMENTS']=credit['MINIMUM_PAYMENTS'].fillna((credit['MINIMUM_PAYMENTS'].mean()))
credit['CREDIT_LIMIT']=credit['CREDIT_LIMIT'].fillna((credit['CREDIT_LIMIT'].median()))

##fit elbow curve
from sklearn.cluster import KMeans

wcss=[]

for i in range(1,11):
    k=KMeans(i,init='k-means++',random_state=54)
    k.fit(credit)
    wcss.append(k.inertia_)
    
import matplotlib.pyplot as plt

plt.plot(range(1,11),wcss)  

##predit the cluster
mat = credit.values
from sklearn.cluster import KMeans
km = KMeans(n_clusters=7)
km.fit(mat)

# Get cluster assignment labels
labels = km.labels_

# Format results as a DataFrame
results = pd.DataFrame([credit.index,labels]).T

##remove the first column since it is s.no
cluster=results[1] 

##give a name to the cluster
cluster=pd.DataFrame(cluster)
cluster.columns=["cluster"]

##combine cluster and data into one data frame
data=pd.concat([credit, cluster], axis=1)

##save the data
data.to_csv('data.csv', index=False)
















    