import pandas as pd


eksik_veriler = pd.read_csv("eksikveriler.csv")
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy = "mean")

Yas = eksik_veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)