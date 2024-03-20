# Importing packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Loading the dataset (Could also pull from API with API key here)
df = pd.read_csv('C:/Users/Dustin Winter/OneDrive/Data/scrap price.csv')

# Specifying variables (dependent and independent)
X = df[['carwidth', 'carheight', 'stroke']]
y = df['price']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the model
model = LinearRegression()

# Fitting the model with the training data
model.fit(X_train, y_train)

# Making predictions using the testing set
y_pred = model.predict(X_test)

# The coefficients
print('Coefficients:', model.coef_)
# The mean squared error
print('Mean squared error:', mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination (R^2):', r2_score(y_test, y_pred))
