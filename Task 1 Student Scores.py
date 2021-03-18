#Task 1 : Prediction using Supervised Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Students Score Dataset.csv')
print("Data imported successfully")
dataset.head(10)

dataset.plot(x = 'Hours', y = 'Scores', style = 'o')
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 1].values 

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print("Training Complete")

line = regressor.coef_*X+regressor.intercept_
plt.scatter(X, y)
plt.plot(X, line);
plt.show()

print(X_test)
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual' : y_test, 'Predicted' : y_pred})
print(df)

Hours = 9.25
Prediction = regressor.predict([[Hours]])
print("Number of Hours", Hours)
print("Predicted Score", Prediction)
