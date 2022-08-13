#Load Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#We use KNeighborsClassifier method
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

#Load Dataset
dataset=pd.read_csv("heart.csv") #if the data set is in a different directory, please provide the file path

print(dataset.head(10)) #First 10 rows
print(dataset.tail(10))
print(dataset.info()) #Give information about dataset
print(dataset.describe()) #istatistiksel hesaplama


print(dataset.dtypes())
print(dataset.shape) #count of rows and columns
print(dataset.columns) #name of columns

print(dataset.isna().values.any()) #Missing Value
#print(dataset.isna().sum()
#fillna() -->Eksik değer doldurmak için..

corr_matris=dataset.corr() #Use corr() function to find the correlation
corr_Features=corr_matris.index
"""
print(corr_matris)
print(corr_Features)
"""
plt.figure(figsize=(10, 12))
p=sns.heatmap(dataset[corr_Features].corr(),annot=True,cmap="RdYlGn")
sns.set_theme(style="darkgrid")
sns.countplot(x="target", data=dataset)
plt.show()

y=dataset['target']
x=dataset.drop(['target'], axis=1) #use drop() function to drop specified labels from rows or columns

KNN_Scores=[]
for K in range(1,21):
    KNN_Classifier=KNeighborsClassifier(n_neighbors=K)
    score=cross_val_score(KNN_Classifier,x,y,cv=10)
    KNN_Scores.append(score.mean())

plt.plot([K for K in range(1,21)], KNN_Scores, color='orange')
for i in range(1,21):
    plt.text(i, KNN_Scores[i - 1], (i, KNN_Scores[i - 1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')
plt.show()

Knn_Classifier=KNeighborsClassifier(n_neighbors=10)
score=cross_val_score(Knn_Classifier,x,y,cv=10)
print(score.mean())



