import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
import sklearn.metrics as skmetrics
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('data_C02_emission.csv')
data =data.drop(["Make", "Model"], axis=1)

#Podijela podataka u skupove a) 
#Numericke varijable ona koje imaju brojeve 
#Kada odabiremo ulazne i izlazne velicine onda kao ulazne velicine stavljamo sve numericke elemente kako bi iz tog skupa mogao uciti, dok za izlazni stavimo jednu 
#velicinu koju cemo trebati dobiti 
#onda pretvorimo data u numpy kako bi bili u obliku "numpy lista"
input_variables = ['Engine Size (L)',
           'Fuel Consumption City (L/100km)',
           'Cylinders',
           'Fuel Consumption Hwy (L/100km)',
           'Fuel Consumption Comb (L/100km)',
           'Fuel Consumption Comb (mpg)']
output_variables=['CO2 Emissions (g/km)']
X=data[input_variables].to_numpy()
y=data[output_variables].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)
#test_size oznacava koliki ce nam postotak biti test skupa ovdje je 20%

#Ovisnost emisije o velicini motora b)
#kao prvu velicinu uzimamo x velicine i na njima opet postavljamo da uzmemo sve retke i samo u ovom slucaju nulti stupac i onda druga velicina bude y 
plt.scatter(x=X_train[:,0], y=y_train, c='b')
plt.scatter(x=X_test[:,0], y=y_test, c='r')

#Standardizacija i histogrami c) 
#minMaxscaler koristimo za skaliranje podataka prije nego što ih stavimo da se model uci 
scaler= MinMaxScaler()
plt.figure()
X_train_n= scaler.fit_transform(X_train)
X_test_n= scaler.transform(X_test)
plt.hist(X_train[:,0])
plt.figure()
plt.hist(X_train_n[:,0])

#Linearna regresija d)
linearModel = lm.LinearRegression()
#napravimo koordinatni sustav sa svim podacima i onda nam je linearna regresija pravac koji je u koordinatnom sustavu 
#pomocu toga pravca odredimo koliko su ukupno sve tocke udaljene od njega sto je udaljenost manja to ce nam se podaci bolje razvrstavati 
linearModel.fit(X_train_n, y_train)
print(linearModel.coef_)

#Procjena izlazne velicine na temelju ulaznih veličina e) 
#nas model koji smo "naucili" i prethodnim koracima koristimo kako bi procjenili koliko je on to tocno odradio 
#uzmemo model i funkciju predict za procjenu i onda iskoristimo skup podataka za testiranje kako bi mogli ocjeniti koliko je nas model tocan 
y_test_p=linearModel.predict(X_test_n)
plt.figure()
plt.scatter(y_test, y_test_p, c='y')
#plt.scatter(x=X_test[:,0], y=y_test, c='k')
#plt.scatter(x=X_test[:,0], y=y_test_p, c='y')

#Vrednovanje modela f)
#ovo su sve metrike koje se mogu iskoristiti za pocjenu tocnosti modela kojeg smo napravili 
#u to ide y_test kao provjera zajedno sa y_test_p sto je bio model koji smo mi naucili 
MSE = skmetrics.mean_squared_error(y_test, y_test_p)
RMSE = math.sqrt(MSE)
MAE = skmetrics.mean_absolute_error(y_test,y_test_p)
MAPE = skmetrics.mean_absolute_percentage_error(y_test,y_test_p)
r2= skmetrics.r2_score(y_test,y_test_p)
print('MSE:', MSE)
print('RMSE', RMSE)
print('MAE:', MAE)
print('MAPE:', MAPE)
print('Koeficijent determinacije:', r2)

#Promjena broja ulaznih vrijednosti g) 
input_variables = [
           'Fuel Consumption City (L/100km)',
           'Fuel Consumption Hwy (L/100km)',
           'Fuel Consumption Comb (L/100km)',
           'Fuel Consumption Comb (mpg)']
output_variables=['CO2 Emissions (g/km)']
X=data[input_variables].to_numpy()
y=data[output_variables].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)
#za skaliranje je moguce iskoristiti i ovaj objekt, ali je bilo predlozeno da se bolje koristi onaj MinMaxScaler 
scaler2 = StandardScaler()
X_train_n= scaler.fit_transform(X_train)
X_test_n= scaler.transform(X_test)
linearModel2 = lm.LinearRegression()
linearModel2.fit(X_train_n, y_train)
y_test_p=linearModel2.predict(X_test_n)
MSE = skmetrics.mean_squared_error(y_test, y_test_p)
RMSE = math.sqrt(MSE)
MAE = skmetrics.mean_absolute_error(y_test,y_test_p)
MAPE = skmetrics.mean_absolute_percentage_error(y_test,y_test_p)
r2= skmetrics.r2_score(y_test,y_test_p)
print('MSE nakon promjene ulaza:', MSE)
print('RMSE nakon promjene ulaza:', RMSE)
print('MAE nakon promjena ulaza:', MAE)
print('MAPE nakon promjene ulaza:', MAPE)
print('Koeficijent determinacije nakon promjene ulaza:', r2)

plt.show()
