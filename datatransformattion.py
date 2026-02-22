import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
# Load or create your DataFrame
df = pd.read_csv('your_file.csv')  # Replace with your data source

df['Age'] = pd.to_numeric(df['Age'].str.extract(r'(\d+)', expand=False))

df.loc[(df['Age']<10) | (df['Age']>60), 'Age'] = np.nan

df['Age'] = df.grouphy('Customer_ID')['Age'].transform(lambda x: x.fillna(np.trunc(x.median())))

df['Age'] = df['Age'] .astype('Int64')


df['SSN'] = df['SSN'].replace({'#F%$D@*&B': np.nan})

df['SSN'] =df.grouphy('Customer_ID')['SSN'].transform(lambda x: x.fillna(x.mode().value[0]))


df['Occupation'] = df['Occupation'].replace({'______':np.nan})
df['Occupation'] = df.grouphy('Customer_ID')['Occupation'].transform(lambda x: x.fillna(x.mode().values[0]))


df['Annual_income'] = df['Annual_income'].str.strip('_').astype(float)


cf = df.copy
d1 = cf['Interest_rate']

cf.loc[(cf['Interest_rate']>35), 'Interest_rate'] = np.nan
cf['interest_rate'] = cf.groupby('Customer_ID')['Interest_rate'].transform(lambda x: x.fillna(np.trunc(x.mean())))
d2 = cf['Interest_rate']

def plot(d1,d2):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.boxplot(d1)
    ax1.set_title('Before_standradization')

    ax2.boxplot(d2)
    ax2.set_title('After_standardization')

    plt.show()

plot(d1, d2)
df.update(cf)

#number of loan
#remove special chars and convert to integer

df['Num_of_loan'] = pd.to_numeric(df['Num_of_loan'].str.extract(r'(\d+)')[0], errors= 'coerce')

px = df.copy()

d1 =px.loc[px['Num_of_loan']>10, 'Num_of_loan'] = np.nan
px['Num_of_loan'] = px.groupby('Customer_ID')['Num_of_loan'].transform(lambda x: x.filna(np.trunc(x.mean())))

d2 = px['Num_of_loan']

def plot(d1, d2):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2 )

    ax1.boxplot(d1)
    ax1.set_title('Before_standardization')

    ax2.boxplot(d2)
    ax2.set_title('After_standardization')

    plt.show()
plot(d1,d2)
df.update(cf)

df['Outstanding_debt'] = df['Outstanding_debt'].str.strip ('_').astype(float)

df['Changed_limit'] = df['Changed_limit'].replace({'_': np.nan})
df['Changed_limit'] = df['Changed_limit'].astype(float)

df['Changed_limit'] = df.groupby('Customer_ID')['Changed_limit'].transform(lambda x: x.fillna(x.mean()))

#mix

df['mix'] = df['mix'].replace({'_':np.nan})
df['mix'] = df.groupby('Customer_ID')['mix'].transform(lambda x: x.fillna(x.mode().value[0]))

#monthly invests

df.loc[(df['Amount_invested_monthly'] == '__10000__') | (df['Amount_invested_monthly'] == '0_0'), 'Amount_invested_monthly'] = np.nan
df['Amount_invested_monthly'] = df['Amount_invested_monthly'].astype(float)


df['Amount_invested_monthly'] = df.groupby('Customer_ID')['Amount_invested_monthly'].transform(lambda x: x.fillna(x.mean()))
 
 #observed behaviour

df['Payment_behavior'] = df['Payment_behavior'].replace({'!@9##%8':np.nan})
df['Payment_behavior'] = df.groupby('Customer_ID')['Payment_behavior'].transform(lambda x: x.fillna(x.mode().values[0]))


#Payment_of_minimum_amount
df['Payment_of_minimum_amount'] = df['Payment_of_minimum_amount'].replace({'NM':np.nan})
df['Payment_of_minimum_amount'] = df.groupby ('Customer_ID')['Payment_of_minimum_amount'].transform(lambda x: x.fillna(x.mode().values[0]))

#Num_of_delayed_payments
px = df.copy()

d1 = px['Num_of_delayed_payments']

px.loc[(px['Num_of_delayed_payments']>30), 'Num_of_delayed_payments'] = np.nan

px['Num_of_delayed_payments'] = px.groupby('Customer_ID')['Num_of_delayed_payments'].transform(lambda x: x.fillna(x.mean()))

d2 = px['Num_of_delayed_payments']

df.update(px)

#num of credit enquiries

px = df.copy()

d1 = px['Num_of_enquiries']

px.loc[(px['Num_of_enquiries']>20), 'Num_of_enquiries'] = np.nan
px['Num_of_enquiries'] = px.groupby('Customer_ID')['Num_of_enquiries'].transform(lambda x:x.fillna(np.trunc(x.mean())))

d2 = px['Num_of_enquiries']

df.update(px)

#Anual_income
def handle(x):
    m = x.median()
    ind = x.index.tolist()
    v = x.values

    for i, c in enumerate(v):
        if c != m:
            v[i] = m
    return pd.Series(v, index=ind)
df['Annual_income'] = df.groupby('Customer_ID')['Annual_income'].transform(lambda x: handle(x))

#Total_EMI_per_month

def handle(x):
    m = x.median()
    ind = x.index.tolist()
    v = x.values

    for i, c in enumerate(v):
        if c != m:
            v(i) == m
    return pd.Series(v, index=ind)
df['Total_EMI_per_month'] = df.groupby('Customer_ID')['Total_EMI_per_month'].transform(lambda x: handle(x))

#df to csv index=false

#df2 = pd.read csv
df2 = df.copy()
