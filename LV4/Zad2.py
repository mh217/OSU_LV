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
#izbacili smo ostale vrijednosti koje nisu numericke 
data =data.drop(["Make", 'Vehicle Class', 'Transmission'], axis=1)


#Iskoristili smo OneHotEncoder za kodiranje vrijednosti buduci da imamo u ulaznim velicinama jednu vrijednost koja nije numericka nego kategoricka 
#kako bi se moglo iskoristiti takvu vrijednost trebas se prebaciti u diskretne numericke vrijednosti 
enc = OneHotEncoder()
input_variables = ['Engine Size (L)',
            'Fuel Consumption City (L/100km)',
           'Cylinders',
           'Fuel Consumption Hwy (L/100km)',
           'Fuel Consumption Comb (L/100km)',
           'Fuel Consumption Comb (mpg)',
           'Fuel Type']
output_variables=['CO2 Emissions (g/km)']
#nominalne kategoricke vrijednosti se kodiraju s K laznih binarnih velicina pri cemu jedna lazna varijabla ima vrijednost 1, a ostale 0 
#napravi se da svaka vrijednost Fule Type bude jedan stupac i u retku kojem je npr. Dizel u stupac dizel se stavlja 1 ostale stupce 0 
#ovo array stavljamo na kraj kako se ti retci ne bi samo nadostekali na ostatak moga dataseta 
X_enc = enc.fit_transform(data[['Fuel Type']]).toarray()
#umjesto trenutnih vrijednosti u datasetu ubacujemo sada ovu novu vrijednost koju smo kodirali u proslom koraku 
#dataset ostaje iste velicine
data['Fuel Type'] = X_enc
#ostatak je relativno isti prebacujemo u tipa numpy kao lista i onda ponovno razdvajamo podatke na isti nacin 
X=data[input_variables].to_numpy()
y=data[output_variables].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#Ponovno smo koristili linearnu regresiju, ali ovoga puta nismo skalirali podatke zato sto to nije bilo zadano zadatkom 
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