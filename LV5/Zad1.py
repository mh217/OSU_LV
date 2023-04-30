import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


#Prikaz podataka za ucenje a) 
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='winter')
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='summer',marker='x')

#Izgraditi logistički model b) 
reg = LogisticRegression()
reg.fit(X_train, y_train)


#Prikaz podatak odluke c) 
coef = reg.coef_[0]
theta0 = reg.intercept_
pravac=(-coef[0]*X_train[:,0]- theta0) /coef[1]
plt.figure()
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='cool')
plt.plot(X_train[:,0], pravac)

#Izracun i prikaz zabune na testnim podacima d) 
y_test_p = reg.predict(X_test)
cm = confusion_matrix(y_test, y_test_p)
print('Matrica zabune:', cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_p))
disp.plot()
print(classification_report(y_test, y_test_p))


#Dobro i lose klasificirani e) 
plt.figure()
plt.scatter(X_test[:,0], X_test[:,1], c=y_test)

plt.figure()
for i in range(len(y_test)):
    if y_test[i] == y_test_p[i] : 
        plt.scatter(X_test[i,0],X_test[i,1], c='k')
    else :
        plt.scatter(X_test[i,0], X_test[i,1],c='y')

plt.show()