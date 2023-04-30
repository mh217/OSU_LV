import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report



labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])


# ucitaj podatke
df = pd.read_csv("penguins.csv")

#Izostaviti vijednosti gdje ih nema 
print(df.isnull().sum())
df = df.drop(columns=['sex'])
df.dropna(axis=0, inplace=True)


# kategoricka varijabla vrsta - kodiranje (object -> int)
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)
print(df.info())



#Postavljanje podataka za ucenje 
output_variable = ['species']
input_variables = ['bill_length_mm',
                    'flipper_length_mm']
X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)




fig = plt.figure()
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)
#Dijagram za primjere za svaku vrstu a)
unq1 = np.unique(y_train, return_counts=True)
unq2 = np.unique(y_test, return_counts=True)
ax1.bar(unq1[0], unq1[1], color=['cyan', 'yellow', 'green'], label=['Adelie','Chinstrap', 'Gentoo'])
ax2.bar(unq2[0], unq2[1], color=['cyan', 'yellow', 'green'], label=['Adelie','Chinstrap', 'Gentoo'])
ax1.set_title('Podaci za ucenje')
ax2.set_title('Podaci za testiranje')
plt.legend()

#Linearna regresija b) 
tran = np.transpose(y_train)[0]
LogisticRegression_model = LogisticRegression()
log_reg = LogisticRegression_model.fit(X_train, tran)

#Atributima ponaci koeficjente c)
t0 = log_reg.intercept_[0]
print(t0)
print(log_reg.coef_) 

#Poziv plot_decision_region d) 
plot_decision_regions(X_train, tran, log_reg)

#Matrica zabune, tocnost i metrike e) 
y_predict = LogisticRegression_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
print("Matrica:", cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_predict))
disp.plot()
print("Tocnost:", accuracy_score(y_test, y_predict))
print(classification_report(y_test, y_predict))
plt.show()



#Dodati još jednu ulazni velicinu f) 
output_variable = ['species']
input_variables = ['bill_length_mm',
                    'flipper_length_mm',
                    'body_mass_g']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()
y=y[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)


LogisticRegression_model2 = LogisticRegression()
log_reg = LogisticRegression_model2.fit(X_train, y_train)
y_predict2 = LogisticRegression_model2.predict(X_test)
cm = confusion_matrix(y_test, y_predict2)
print("Matrica:", cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_predict2))
disp.plot()
print("Tocnost:", accuracy_score(y_test, y_predict2))
print(classification_report(y_test, y_predict2))
plt.show()
