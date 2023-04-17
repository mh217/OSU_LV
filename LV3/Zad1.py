import pandas as pd
import numpy as np
import matplotlib.pylab as plt 

data =pd.read_csv('data_C02_emission.csv')


#Koliko ih sadrzi, tipovi, izostale i duplicirane vrijednost, kategoricke velicine a) 
print('Mjerenja:', len(data))
print('Tip svake velicine:', data.info())

izostali = data.isnull().sum()
print('Izostali:', izostali)
if sum(izostali) != 0: 
    data.dropna(axis=0)
    data.dropna(axis=1)
data = data.reset_index(drop=True)


data.drop_duplicates()
data = data.reset_index(drop=True)
print(len(data))

data[data.select_dtypes(['object']).columns] = data.select_dtypes(['object']).apply(lambda x : x.astype('category'))
print(data.info())

#Najveca i najmanja potrosnja u gradu b)
print('Najvece:')
print(data.nlargest(3, 'Fuel Consumption City (L/100km)')[['Make', 'Model', 'Fuel Consumption City (L/100km)']])
print('Najmanje:')
print(data.nsmallest(3, 'Fuel Consumption City (L/100km)')[['Make', 'Model', 'Fuel Consumption City (L/100km)']])


#Motor izmedu 2.5 i 3.5 L i prosjecna CO2 emisije c)
sorted = data[(data['Engine Size (L)']>=2.5) & (data['Engine Size (L)']<=3.5)]
print('Velicina motora izmedu 2.5 i 3.5:', len(sorted))
print('Prosjecna CO2 emisija:', sorted[['CO2 Emissions (g/km)']].mean())

#Samo Audi i sa 4 cilindra i prosjecna emisija CO2 d) 
sorted_audi = data[data['Make'] == 'Audi']
print('Samo Audi:', len(sorted_audi))
samo4cilindra=sorted_audi[sorted_audi['Cylinders'] == 4]
print('Audi sa 4 cilindra', len(samo4cilindra))
print('Prosjecna emisija audia:' , samo4cilindra[['CO2 Emissions (g/km)']].mean())

#Parni broj cilindara e)
fourcylinders = data[data['Cylinders']==4]
sixcylinders = data[data['Cylinders']==6]
eightcyliners = data[data['Cylinders']==8]
tencyliners = data[data['Cylinders']==10]
twcyliners = data[data['Cylinders']==12]
stcyliners = data[data['Cylinders']==16]
print('Emisija CO2 sa 4 cilindra', fourcylinders['CO2 Emissions (g/km)'].mean())
print('Emisija CO2 sa 6 cilindra', sixcylinders['CO2 Emissions (g/km)'].mean())
print('Emisija CO2 sa 8 cilindra', eightcyliners['CO2 Emissions (g/km)'].mean())
print('Emisija CO2 sa 10 cilindra', tencyliners['CO2 Emissions (g/km)'].mean())
print('Emisija CO2 sa 12 cilindra', twcyliners['CO2 Emissions (g/km)'].mean())
print('Emisija CO2 sa 16 cilindra', stcyliners['CO2 Emissions (g/km)'].mean())

#Potrosnja kod dizela i kod benzina f)
dizel = data[data['Fuel Type'] == 'D']
benzin = data[data['Fuel Type']== 'X']
print('Dizel:', dizel[['Fuel Consumption City (L/100km)']].mean()) 
print('Benzin', benzin[['Fuel Consumption City (L/100km)']].mean())
print('Medijan dizel:', dizel[['Fuel Consumption City (L/100km)']].median()) 
print('Medijan benzin', benzin[['Fuel Consumption City (L/100km)']].median())

#Ima 4 cilindra, dizel i najveca gradska potrosnja g)
car = data[(data['Cylinders']==4) & (data['Fuel Type']=='D')]
print('Najveca gradska potrosnja od dizela:', car[['Fuel Consumption City (L/100km)']].max())

#Rucni mjenjac h) 
mjenjac = data[data['Transmission'].str.startswith('M')]
print('Rucni mjenjac', len(mjenjac))

#Korelacija numerickih velicina i) 
print(data.corr(numeric_only =True))
#Sto je broj veci to znaci da je bolje korelacija izmedu dva elemenata, na dijagonalama su jedinice zato sto svaki sa sobom ima najvecu korelaciju
