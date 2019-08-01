import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
import category_encoders as ce
import matplotlib.pyplot as plt
import scikitplot as skplt

# Set display width
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 50)

# Load data
data = pd.read_csv('all_seasons.csv')

# Preprocessing
df = pd.DataFrame(data)
#df.to_csv(r"D:\University of Waterloo\Thrid Semester\ECE657\Final\OverwatchAnalysis-ZackTack\df.csv")
df = df.iloc[276:,]
df.replace(to_replace='P',value=0,inplace=True)
drop_column_list = ['Game #','Start SR','End SR','SR Change','Team SR avg','Enemy SR avg',
                    'season','Death_career','Heal_career','Elim_career','Obj_kills_career',
                    'Obj_time_career','Dmg_career']
for i in drop_column_list:
    df = df.drop(columns=i)
df = df.fillna(df.mode().iloc[0])
print(df.describe())
print(df.dtypes)
print(df.head())

# Split dataset into training and testing sets
X= df.drop(columns='Result')
Y = df['Result'].copy()
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=20)

# Convert categorical data to numerical
ohe = ce.OneHotEncoder(use_cat_names=True)
X_train_ohe = ohe.fit_transform(X_train)
X_test_ohe = ohe.transform(X_test)

# Training MLP
clf = MLPClassifier(solver='lbfgs',activation='logistic',learning_rate='constant',max_iter=600,hidden_layer_sizes=(10,10))
clf.fit(X_train_ohe,Y_train)
prediction = clf.predict(X_test_ohe)

# Evaluate MLP
print(clf.get_params())
print('The accuracy on test set is: ',clf.score(X_test_ohe,Y_test))
print(classification_report(Y_test,prediction))
skplt.estimators.plot_learning_curve(clf,X_train_ohe,Y_train)
plt.show()