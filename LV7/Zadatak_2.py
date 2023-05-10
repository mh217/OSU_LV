import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
for i in range(1,6) : 
    img = Image.imread("imgs\\test_"+ str(i) +".jpg")

    # prikazi originalnu sliku
    plt.figure()
    plt.title("Originalna slika")
    plt.imshow(img)
    plt.tight_layout()
    plt.show()

    # pretvori vrijednosti elemenata slike u raspon 0 do 1
    img = img.astype(np.float64) / 255

    # transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))

    # rezultatna slika
    img_array_aprox = img_array.copy()

    #Primjena algoritma K srednjih vrijednosti 
    km = KMeans(n_clusters=4, init='random', n_init=5, random_state=0)
    lable = km.fit_predict(img_array_aprox)
    rgb_cols = km.cluster_centers_.astype(np.float64)
    img_quant = np.reshape(rgb_cols[lable], (w,h,d))

    plt.figure()
    plt.imshow(img_quant)
    plt.show()

    #Lakat metoda 
    distortions =[]
    K=range(1,10)
    for k in K :
        kmm = KMeans(n_clusters=k, init='random', n_init=5, random_state=0)
        kmm.fit(img_array_aprox)
        distortions.append(kmm.inertia_)

    plt.figure()
    plt.plot(K,distortions)
    plt.show()

    #Binarna slika 
    for i in range(4) :
        binar = lable ==[i]
        new_image = np.reshape(binar, (img.shape[0:2]))
        new_image = new_image*1
        x=int(i/2)
        y=i%2
        plt.imshow(new_image)


    plt.show()

