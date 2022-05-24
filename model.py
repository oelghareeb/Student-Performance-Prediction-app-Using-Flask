
### Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle

warnings.filterwarnings('ignore')

### Import Datset
df = pd.read_csv("data_breast-cancer-wiscons.csv")
# we change the class values (at the column number 2) from B to 0 and from M to 1
df.iloc[:,1].replace('B', 0,inplace=True)
df.iloc[:,1].replace('M', 1,inplace=True)

### Splitting Data

X = df[['radius_mean', 'texture_mean', 'area_mean', 'concavity_mean',
       'concave points_mean', 'area_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'concavity_worst',
       'concave points_worst', 'symmetry_worst']]
y = df['diagnosis']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=0)

#### Data Preprocessing

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)


##
from sklearn.svm import SVC
clf_svm = SVC(probability=True)
clf_svm.fit(x_train, y_train)
predictions = clf_svm.predict(x_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


print("Confusion Matrix : \n\n" , confusion_matrix(predictions,y_test))

print("Classification Report : \n\n" , classification_report(predictions,y_test),"\n")


pickle.dump(clf_svm, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(model)