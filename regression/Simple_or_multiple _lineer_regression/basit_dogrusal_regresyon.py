

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('satislar.csv')

aylar = veriler[["Aylar"]]

satislar = veriler[["Satislar"]]

# print(satislar)
# print(aylar)
# print(veriler)


#verşlerin egitim ve test olarak bolunmesi

from sklearn.model_selection import train_test_split

x_train , x_test , y_train,y_test= train_test_split(aylar,satislar,test_size=0.33,random_state=0)

'''
#oznitelik olcekleme
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train =  sc.fit_transform(y_train)
Y_test =  sc.fit_transform(y_test)
'''
# print(x_test)
# print(x_train)
# print(y_test)
# print(y_train)

#mmodel inşasi (lineer regression)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
tahmin = lr.predict(x_test)
print(tahmin)
print(y_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()



plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))

plt.title("aylara gore satis")
plt.xlabel("aylar")
plt.ylabel("satislar")

plt.show()