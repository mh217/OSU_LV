import pandas as pd
import matplotlib . pyplot as plt
import numpy as np

data =pd.read_csv('data_C02_emission.csv')

#Emisija CO2 plinova a) 
plt.figure()
data['CO2 Emissions (g/km)'].plot(kind = 'hist')
plt.title('CO2 Emissions (g/km)')

#Odnos gradske potrošnje i emisije CO2 b) 
plt.figure()
lables, index = np.unique(data['Fuel Type'], return_inverse=True)
ax = plt.subplot()
sc= ax.scatter(data['CO2 Emissions (g/km)'],data['Fuel Consumption City (L/100km)'], c=index, alpha=0.8)
plt.xlabel('CO2 emissions (g/km)')
plt.ylabel('Fuel Consumption City (L/100km)')
ax.legend(sc.legend_elements()[0], lables)

#Razdioba izvangradske potrošnje s obziron na tip goriva c) 
data.boxplot(column = ['Fuel Consumption Hwy (L/100km)'], by =['Fuel Type'])


#Broj vozila po tipu goriva d) 
fig=plt.figure()
ax1=fig.add_subplot(121)
ax1.set_title('By Fuel Type')
data_group = data.groupby('Fuel Type').size()
data_group.plot(kind='bar')

#Prosjecna CO2 emisija s obzirom na broj cilindara e) 
ax2=fig.add_subplot(122)
ax2.set_title('By Cylinders')
data_group2 = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
data_group2.plot(kind='bar')


plt.show()


