# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
# veri yukleme
veriler = pd.read_csv('maaslarYeni.csv')

x = veriler.iloc[:,2:3]
y = veriler.iloc[:,5:]
X = x.values
Y = y.values


#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())


#polynomial regression


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)


#tahminler
print("poly ols")
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())


#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()

x_olcekli = sc1.fit_transform(X)

sc2=StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

#support vector regressyon
from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

# plt.scatter(x_olcekli,y_olcekli,color='red')
# plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')
# plt.show()


print("svr ols")
model3 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())


#decision tree regression
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

# plt.scatter(X,Y,color = "red")
# plt.plot(X,r_dt.predict(X),color="blue")
# plt.show()

print(r_dt.predict([[11]]))
print("dt ols")
model4 = sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())


#ranom forest regressyon
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)#estimator = kac farkli decision tree cizilsin

rf_reg.fit(X,Y.ravel())
print(rf_reg.predict([[6.5]]))

print("\n\nrf ols")
model5 = sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())




from sklearn.metrics import r2_score

print("random forest r2 degeri")
print(r2_score(Y,rf_reg.predict(X)))



#ozet R2 degerleri
print("ozet olarak r2 degerleri asagidadir")
print("linear R2 degeri")
print(r2_score(Y,lin_reg.predict(X)))


print("polynomial R2 degeri")
print(r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(X))))

print("suport vector regression r2 degeri")
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

print("decision tree r2 degeri")
print(r2_score(Y,r_dt.predict(X)))

print("random forest r2 degeri")
print(r2_score(Y,rf_reg.predict(X)))
