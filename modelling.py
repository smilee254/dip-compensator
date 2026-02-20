import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE

from sklearn.neighbours import KNeighborsClaasiifier as KNN
from sklearn.ensemble import RandomFrestClassifier as RFC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix as cm, accuracy_score, precision_score, recall_score, f1_score

#transfor the data values into a standard form
df3 = df2.copy().drop(columns=['Customer_ID', 'ID'])

for c in df3.columns:
    df3[c] = LabelEncoder().fit_transform(df3[c])

y = df3.pop('Credit_score')
x = df3