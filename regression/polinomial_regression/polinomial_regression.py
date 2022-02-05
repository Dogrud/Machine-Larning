
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


veriler = pd.read_csv("maaslar.csv")

#data frame dilimleme(slicing)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#numpyarray donusumu
X = x.values
Y = y.values
#print(x)
#print(y)

#lineer regression
#dogrusal model olusturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)


#2.dereceden polynomial regression
#dogrusal olmayan (non lineer) model olusturma
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2 )
x_poly = poly_reg.fit_transform(X)
lin_reg2  = LinearRegression()
lin_reg2.fit(x_poly,y)


#4. dereceden polynomial regression
#dogrusal olmayan (non lineer) model olusturma
poly_reg3 = PolynomialFeatures(degree = 4 )
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3  = LinearRegression()
lin_reg3.fit(x_poly3,y)

# Gorsellestirme

plt.scatter(X,Y,color="red")
plt.plot(x,lin_reg.predict(X),color  ="blue")
plt.show

plt.scatter(X,Y,color ="red")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color="blue")
plt.show()

plt.scatter(X,Y,color ="red")
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)),color="blue")
plt.show()

#tahminler
print(lin_reg.predict(11))
print(lin_reg.predict(6.6))

print(lin_reg2.predict(poly_reg.fit_transform(11)))
print(lin_reg2.predict(poly_reg.fit_transform(6.6)))