# GOOGLE COLAB LINK : https://colab.research.google.com/drive/1-MkKHWGOD_GB3Gqcnaxst-4UoyDm7EtT?usp=sharing
# DATASET FROM KAGGLE: https://www.kaggle.com/ealaxi/paysim1/download

# Importing Modules
import pandas as pd

#Importing data
dataset = '/content/drive/MyDrive/Datasets/Payment_fraud_dataset.csv'
df = pd.read_csv(dataset)

#First we check columns 
columns = df.columns
columns

# Checking Transaction types
df.type.value_counts()

correlation = df.corr()
print(correlation['isFraud'].sort_values(ascending=False))

# We change the tranfer methods into integers so it became easy to feed into our model

df['type'] = df['type'].map({'CASH_OUT': 1 , 'PAYMENT':2 ,
                             'CASH_IN': 3, 'TRANSFER':4,
                             'DEBIT':5})

# changing fraud  into 0 or 1 if 0 means no fraud or 1 means fraud
df['isFraud'] =df['isFraud'].map({0:'No Fraud', 1:'Fraud'})
print(df.head())

# Training Model
from sklearn.model_selection import train_test_split
import numpy as np
# We take only those field which are necessary for our model 
x = np.array(df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']])
y = np.array(df[['isFraud']])

# Using DecisionTreeClassifier to predict 
from sklearn.tree import DecisionTreeClassifier
# splitting data
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.20, random_state=42)
model = DecisionTreeClassifier()
# Training Model
model.fit(xtrain, ytrain)

# Checking accuracy rate of our model
model.score(xtest, ytest)

# prediction
features = np.array([[4,9000.60,9000.60,0.0]])
model.predict(features)
