
### Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle

warnings.filterwarnings('ignore')

### Import Datset
df = pd.read_csv("student_data.csv")

### Splitting Data

X = df[['absences', 'G1', 'G2']]
y = df[['G3']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

#### Data Preprocessing

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)


##
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=1000)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)


from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error

print("R2 Score :",r2_score(y_test,y_pred))
print("Mean Squared Error :",mean_squared_error(y_test, y_pred))
print('Mean Absolute Error :', mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error :', np.sqrt(mean_squared_error(y_test, y_pred)))


pickle.dump(rf_reg, open('SP_rf.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

model = pickle.load(open('SP_rf.pkl', 'rb'))
print(model)