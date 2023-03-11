import numpy as np 
import matplotlib.pyplot as plt 

img=plt.imread("road.jpg")
img=img[:,:,0].copy()
print(img.shape)

#Posijetliti sliku a)
plt.imshow(img, vmin=0, vmax=77, cmap="gray")
plt.show()

#Prikaz druge cetvrtine slike b)
cropped=img[0:,160:320]
plt.imshow(cropped, cmap="gray")
plt.show()

#Rotacija slike za 90 c)
plt.figure()
plt.imshow(np.rot90(img,3), cmap="gray")
plt.show()


#Zrcaljenje slike d)
plt.imshow(np.fliplr(img), cmap="gray")
plt.show()