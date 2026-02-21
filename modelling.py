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

# data resampling
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(x,y)

# split into train and testing data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, shuffle=True, stratify=y_resampled)

#Models

models = {
    'K_Nearest_Neighbour': KNN(n_neighbors=5, weights='distance', algorithm='ball_tree', leaf_size=30),
    'Decision_Tree': DTC(criterion='entropy', max_depth=40, splitter='best', random_state=42, max_features='sqrt'),
    'Random_forest': RFC(n_estimators=150, criterion='gini', max_features='sqrt', random_state=42, max_depth=30, class_weight='balanced_subsample')
}