import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
#z = np.reshape(y,(-1,1))
y = y.reshape(len(y),1)

# To calculate different mean and standard deviation for both X and y and use the
# relevant one with each
sc_X = StandardScaler()
sc_y = StandardScaler()

X_standard = sc_X.fit_transform(X)
y_standard = sc_y.fit_transform(y)

print("X: ",X_standard)
print("y: ",y_standard)

# model
regressor = SVR(kernel='rbf')
regressor.fit(X_standard,y_standard)