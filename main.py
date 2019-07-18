import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
import numpy as np

data = pd.read_csv('all_seasons.csv')

df = pd.DataFrame(data).dropna()
df.replace(to_replace='P',value=0,inplace=True)
df = df.drop(columns='Game #')

X= df.drop(columns='Result')
Y = df['Result'].copy()
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=40)

ohe = ce.OneHotEncoder(use_cat_names=True)
X_train_ohe = ohe.fit_transform(X_train)
X_test_ohe = ohe.transform(X_test)
ord = ce.OrdinalEncoder()
Y_train_od = ord.fit_transform(Y_train)
Y_test_od = ord.transform(Y_test)
print(X_train_ohe,X_test_ohe,Y_train_od,Y_test_od)


clf = MLPClassifier()
clf.fit(X_train_ohe,Y_train_od)
prediction = clf.predict(X_test_ohe)
prediction = pd.DataFrame(prediction)
#print(clf.score(prediction,Y_test_od))
