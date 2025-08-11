# Credit Card Fraud Detection - K-Nearest Neighbor(KNN)

## Importing the Dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv(r'C:\Users\subha\OneDrive\Desktop\project\Creditcard\creditcard.csv')

# First 5 rows of the dataset
print(credit_card_data.head())

# Dataset description
print(credit_card_data.describe().transpose())

# Dataset information
print(credit_card_data.info())

# Checking the number of missing values in each column
print(credit_card_data.isnull().sum())

# Histogram of all features
credit_card_data.hist(figsize=(20, 20))
plt.show()

## Standardize the variables
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(credit_card_data.drop(["Class"], axis=1)))
y = credit_card_data["Class"]

print(X.head())

## Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

## Using KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

## Predictions and Evaluations
from sklearn.metrics import classification_report, confusion_matrix

print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))
