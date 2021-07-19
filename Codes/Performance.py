# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 16:51:40 2020

@author: shovon5795
"""

import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

dataset = pd.read_csv(r"C:\Users\shovon5795\Desktop\Research\Mamun sir saha reno\ABCD.csv")
b=[]
c=[]
d=[]

df = dataset.T
X = df.iloc[0,:].values

for i in range(1,4):        
    Y = df.iloc[i,:].values
    b.insert(i, adjusted_rand_score(X, Y)) 
    c.insert(i, normalized_mutual_info_score (X, Y))
    d.insert(i, adjusted_mutual_info_score(X, Y))

