import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler



import seaborn as sns # for statistical data visualization
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("https://raw.githubusercontent.com/harika-bonthu/SupportVectorClassifier/main/datasets_229906_491820_Fish.csv")
#print(df.isnull().sum())
df=df.sample(frac=1)


features=df.drop(['Species'],axis='columns')
target=df[['Species']]
#target=df.Species # Second way

X_train,X_test,Y_train,Y_test=train_test_split(features,target,test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model=SVC(kernel='linear',C=6)

model.fit(X_train,Y_train)

#svm_pred=model.predict(X_test)

acc=model.score(X_test,Y_test)


print('Training set score: {:.4f}'.format(model.score(X_train, Y_train)))

print('Test set score: {:.4f}'.format(model.score(X_test, Y_test)))