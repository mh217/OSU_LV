import numpy as np 
import matplotlib.pyplot as plt 

crno=np.ones((50,50))
bijelo=np.zeros((50,50))

stack1=np.vstack((crno,bijelo))
stack2=np.vstack((bijelo,crno))
img=np.hstack((stack2,stack1))
plt.imshow(img, cmap="gray")
plt.show()