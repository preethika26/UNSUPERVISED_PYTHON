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

##DIMENSIONALITY REDUCTION USING PRONICIPLE COMPONENT ANALYSIS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


x=pd.DataFrame(credit)
x=sc.fit_transform(x)
x=pd.DataFrame(x)

from sklearn.decomposition import PCA
pca=PCA()
x=pca.fit_transform(x)

res=pca.explained_variance_ratio_*100  ##to see in % multiply by 100
a=np.cumsum(pca.explained_variance_ratio_*100) ##cumulative sum

##SCREE PLOT

scores=pd.Series(pca.components_[0])
scores.abs().sort_values(ascending=False)

var=pca.components_[0]
plt.bar(x=range(1,len(var)+1),height=res)
plt.show()

##remove the variables according to PCA
crt_data=pd.DataFrame(credit)
c=crt_data.drop(['INSTALLMENTS_PURCHASES','CASH_ADVANCE_FREQUENCY','PAYMENTS','PURCHASES_INSTALLMENTS_FREQUENCY','PRC_FULL_PAYMENT'],axis=1)

##fit elbow curve
from sklearn.cluster import KMeans

wcss=[]

for i in range(1,11):
    k=KMeans(i,init='k-means++',random_state=54)
    k.fit(c)
    wcss.append(k.inertia_)
    
import matplotlib.pyplot as plt
plt.plot(range(1,11),wcss)  

##pipelines combine scaler and kmeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler=StandardScaler()
km = KMeans(n_clusters=7)

from sklearn.pipeline import make_pipeline
pipeline=make_pipeline(scaler,km)

##predict the cluster
pred=pipeline.fit_predict(c)

##convert the predicted to dataframe
pred=pd.DataFrame(pred)

##name the column as cluster
pred.columns=["cluster"]

##combine cluster and data into one data frame
data=pd.concat([c, pred], axis=1)







    