import numpy as np
import matplotlib.pyplot as plt

fig=plt.figure()
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)

arra= np.loadtxt('data.csv', delimiter=',', skiprows=1, dtype=float)

#Na koliko je osoba izvršeno mjerenje a)
size=len(arra)
print('Izvrseno je mjenjenja na:', size)
size=int(size)

#Odnos visine i tezine b)
x=np.array(arra[:,1])
y=np.array(arra[:,2])
ax1.set_title('Odnos svih visina i tezina')
ax1.scatter(x,y,alpha=0.5,c='b', s=5)

#Za svaku 50 osobu c)
x1=x[::50]
y1=y[::50]
ax2.set_title('Odnos visine i tezine svake 50 osobe')
ax2.scatter(x1,y1,alpha=0.5,c='k', s=5)
plt.show()

#Min,Max i srednja vrijednost d)
print('Min', x.min())
print('Max:', x.max())
print('Srednja vrijednost:', x.mean())


#Min,Max i srednje vrijednost po spolovima e)
male=(arra[:,0]==1)
female=(arra[:,0]==0)

print('Min muski:', arra[male,1].min())
print('Max muski:', arra[male,1].max())
print('Srednja vrijednost muski:', arra[male,1].mean())

print('Min zene:', arra[female,1].min())
print('Max zene:', arra[female,1].max())
print('Srednja vrijednost zene:', arra[female,1].mean())











