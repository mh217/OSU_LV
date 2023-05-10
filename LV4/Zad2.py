import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm
import sklearn.metrics as skmetrics

data = pd.read_csv('data_C02_emission.csv')
data =data.drop(["Make", 'Vehicle Class', 'Transmission'], axis=1)


enc = OneHotEncoder()
input_variables = ['Engine Size (L)',
            'Fuel Consumption City (L/100km)',
           'Cylinders',
           'Fuel Consumption Hwy (L/100km)',
           'Fuel Consumption Comb (L/100km)',
           'Fuel Consumption Comb (mpg)',
           'Fuel Type']
output_variables=['CO2 Emissions (g/km)']
X_enc = enc.fit_transform(data[['Fuel Type']]).toarray()
labels = np.argmax(X_enc, axis=1)
data['Fuel Type'] = labels
X=data[input_variables].to_numpy()
y=data[output_variables].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

linearModel = lm.LinearRegression()
linearModel.fit(X_train, y_train)
print(linearModel.coef_)
y_test_p = linearModel.predict(X_test)
plt.figure()
plt.scatter(x = y_test,y = y_test_p, c = 'b')
plt.plot(y_test,y_test, color='green')


#Izracunavanje maksimalne pogreske 
abs = abs(y_test - y_test_p)
max = np.argmax(abs)
print(abs[max])
print(data.at[data.index[max], 'Model'])


plt.show()