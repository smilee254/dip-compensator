import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('default')

from string import punctuation
from math import trunc

df = pd.read_csv('kaggle/input/credit-score-classication/train.csv' , low_memory=False)
df . info()

df . isna () . sum()


lookup = (df.dropna(subset=['Name']). drop_duplicates(subset=['Customer_ID']).set_index('Customer_ID')['Name'])

df['Name'] = df['Customer_ID'].map(lookup).fillna(df['Name'])

lookup = (df.dropna(subset=['Monthly_inhand_salary']).drop_duplicates(subset='Customer_ID').set_index('Customer_ID')['Monthly_inhand_salary'])

loan_types = list()
def types(x):
    if type(x) == list:
        for i in x:
            i = i.lstrip(' and ')
            if i not in loan_types:
                loan_types.append(i)

df['Type_of_loan'].str.split( ' .' ).appply(lambda x: types(x))

ax = df.loc [(df['Type_of_loan'].isna())].drop_duplicates(subset='Customer_ID')

values = set(ax['Customer_ID'])
cx = df['Customer_ID'].isin(values).astype(object)

df['Type_of_loan'].fillna(value='Not specified', inplace=True)

def rem_xtr(x):
    if type(x) == list:
        for c in x:
            if c in list(punctuation):
                x.remove(c)
    return int(''.join(str(c)for c in x).strip())

df['Num_of_delayed_payments'] = df['Num_of_delayed_payments'].str.split('').apply(lambda x: rem_xtr(x))

df['Num_of_delayed_payments'] = df['Num_of_delayed_payments']. fillna((df.grouphy('Customer_ID')['Num_of_delayed_payments'].transform('median')))


df['Num_credit'] = df['Num_credit'].fillna(df.groupby('Customer_ID')['Num_credit'].transform('median'))

MONTHS = {'January': 0, 'February': 1, 'March': 2, 'April': 3, 'May': 4, 'June': 5, 'July': 6,
          'August': 7}

first_month = (df.dropna(subset='Credit_hist_age').drop_duplicates(subset='Customer_ID')[['Customer_ID' 'Month', 'Credit_hist_age']])
null_month =  (df.loc[df['Credit_hist_age'].isna()][['Customer_ID', 'Month', 'Credit_hist_age']])

def fill(x):
    nm = null_month ['Month'].loc[(null_month['Customer_ID'] == x) & (null_month['Credit_hist_age'].isna ())].tolist()
    fm = first_month.loc[first_month['Customer_ID'] == x] ['Month'].tolist().pop()

    cs = first_month.loc[first_month['Customer_ID'] == x] ['Credit_hist_age'].tolist().pop()

    (y, m) = cs.strip('Month').split('years and')
    (y, m) = (int(y), int(m))

    for i in nm:
        if MONTHS.get(i) > MONTHS.get(fm):
            new_m = m + MONTHS.get(i)
            if new_m >= 12:
                new_m = new_m - 12
                y = y + 1

    else:
        new_m = m - (MONTHS.get(i) + MONTHS.get(fm))
        if new_m < 0:
            new_m = 12 + new_m
            y = y - 1
    v = str(y)+'years and' +str(new_m)+ 'Month'
    null_month.loc[(null_month['Customer_ID'] == x) & (null_month['Month'] == i), 'Credit_hist_age'] = v

    null_month ['Customer_ID'].map(lambda x: fill(x))

    df.update(null_month)


    ex = df.loc [df['Amout_invested'].isna()].tolist()

    plt.figure(figsize=(16, 8))
    sns.barplot(data=ex, y='Cucstomer_ID', x = 'count')
    plt.show()

    df.dropna(subset='Amount_invested_monthly',inplace = True)


    df.dropna(subset='Monthly_balance', inplace = True)

