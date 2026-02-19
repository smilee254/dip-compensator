import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

plt.style.use('default')

# Replace with actual file path
csv_path = 'your_file.csv'  # Update this to your CSV file path
try:
    df2 = pd.read_csv(csv_path)
    df2.info()
except FileNotFoundError:
    print(f"Error: File '{csv_path}' not found. Please check the path.")
    exit()

# Check for required columns
required_columns = ['Customer_ID', 'Age', 'Occupation', 'Annual_income', 'Monthly_inland_salary']
missing_columns = [col for col in required_columns if col not in df2.columns]
if missing_columns:
    print(f"Error: Missing columns in CSV: {missing_columns}")
    exit()

dx = df2.copy()
le = LabelEncoder()
for c in dx.columns:
    if dx[c].dtype == 'object':  # Only encode categorical columns
        dx[c] = le.fit_transform(dx[c])

plt.figure(figsize=(30, 17))
sns.heatmap(dx.corr(), linewidths=5, annot=True, vmax=1, vmin=0)
plt.show()

# Age_distribution
ax = df2.groupby('Customer_ID')['Age'].median()

plt.figure(figsize=(12, 8))
sns.histplot(data=ax, binwidth=2, kde=True)
plt.title('Age_distribution_among_users')
plt.show()

# Total annual income in a given occupation
cx = df2.groupby('Occupation')['Annual_income'].sum().reset_index().sort_values(by='Annual_income', ascending=False)

plt.figure(figsize=(20, 8))
sns.barplot(data=cx, x='Occupation', y='Annual_income', palette='muted')
plt.title('Total annual income for each occupation')
plt.show()

# Total monthly inland salary in a given occupation
cx = df2.groupby('Occupation')['Monthly_inland_salary'].sum().reset_index().sort_values(by='Monthly_inland_salary', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(data=cx, x='Occupation', y='Monthly_inland_salary', palette='muted')
plt.title('Total monthly inland salary in a given occupation')
plt.show()
