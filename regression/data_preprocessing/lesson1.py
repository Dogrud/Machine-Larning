
#ders 6 kutuphanelerin yuklenmesi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#kod bolumu
#veri yukleme

veriler = pd.read_csv("veriler.csv")


boy=veriler[["boy"]]
print(boy)

boykilo = veriler[["boy","kilo"]]
print(boykilo)

class insan:
    boy =80

ali = insan()
print(ali.boy)
    




