import pandas as pd
import numpy as np


veriler = pd.read_csv("odev_tenis.csv")




#ecoder : kategorik -> numeric



from sklearn import preprocessing

le = preprocessing.LabelEncoder()

veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)
print(veriler2)
outlook= veriler2.iloc[:,0:1].values

ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()
print(outlook)
#windy = ohe.fit_transform(windy).toarray()
#play = ohe.fit_transform(play).toarray()



sonuc=pd.DataFrame(data=outlook,index=range(14),columns = ["o","r","s"] )
son_veriler = pd.concat([sonuc,veriler.iloc[:,1:3]],axis = 1)
son_veriler = pd.concat([veriler2.iloc[:,-2:],son_veriler],axis = 1)
print(son_veriler)



#verşlerin egitim ve test olarak bolunmesi
from sklearn.model_selection import train_test_split

x_train , x_test , y_train,y_test= train_test_split(son_veriler.iloc[:,:-1],son_veriler.iloc[:,-1:],test_size=0.33,random_state=0)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train , y_train)

y_pred = regressor.predict(x_test)

print(y_pred)
print(y_test)

#backward elimination (geriye eleme yontemi ile eleme p value and significant number)
import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int) , values = son_veriler.iloc[:,:-1],axis = 1)

X_l = son_veriler.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS( son_veriler.iloc[:,-1:],X_l).fit()
#print(model.summary())

son_veriler = son_veriler.iloc[:,1:]
X = np.append(arr = np.ones((14,1)).astype(int) , values = son_veriler.iloc[:,:-1],axis = 1)

X_l = son_veriler.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS( son_veriler.iloc[:,-1:],X_l).fit()
#print(model.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train , y_train)

y_pred = regressor.predict(x_test)
print(y_pred)