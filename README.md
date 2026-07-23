# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Niranjan V
RegisterNumber:  212224110042

# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1) Load dataset
df = pd.read_csv('exp_2_dataset_student_scores.csv')   # CSV should have two columns, e.g. "Hours","Scores"
print("First 5 rows:\n", df.head(), "\n")
print("Last 5 rows:\n", df.tail(), "\n")

# 2) Prepare input (X) and output (Y)
# Assume CSV columns: Hours (feature) and Scores (target)
X = df.iloc[:, :-1].values   # all rows, all columns except last -> shape (n_samples, 1)
Y = df.iloc[:, -1].values    # all rows, last column -> shape (n_samples,)
print("X (features):", X.flatten())
print("Y (targets):", Y)

# 3) Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
print("\nTraining samples:", len(X_train), " Testing samples:", len(X_test))

# 4) Create and train the model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)   # fit on training data

# 5) Predict on the test set
Y_pred = regressor.predict(X_test)
print("\nPredicted values:", np.round(Y_pred, 2))
print("Actual values   :", Y_test)

# 6) Plot training results
plt.figure(figsize=(6,4))
plt.scatter(X_train, Y_train, color="orange", label="Training data")
plt.plot(X_train, regressor.predict(X_train), color="red", label="Fitted line")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()

# 7) Plot testing results (use X_test sorted for a nicer line)
order = np.argsort(X_test.flatten())
X_test_sorted = X_test.flatten()[order]
Y_test_sorted = Y_test[order]
Y_pred_sorted = Y_pred[order]
plt.figure(figsize=(6,4))
plt.scatter(X_test, Y_test, color="blue", label="Test data")
plt.plot(X_test_sorted, Y_pred_sorted, color="green", label="Predictions")
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()

# 8) Evaluation metrics
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

# 9) Example: predict for new students
new_hours = np.array([[2.5], [8.0]])   # shape must be (n_samples, 1)
pred_new = regressor.predict(new_hours)
print("\nPredictions for new hours", new_hours.flatten(), "=>", np.round(pred_new,2))
*/
```

## Output:
```
First 5 rows:
    Hours  Scores
0    2.5      21
1    5.1      47
2    3.2      27
3    8.5      75
4    3.5      30 

Last 5 rows:
     Hours  Scores
20    2.7      30
21    4.8      54
22    3.8      35
23    6.9      76
24    7.8      86 

X (features): [2.5 5.1 3.2 8.5 3.5 1.5 9.2 5.5 8.3 2.7 7.7 5.9 4.5 3.3 1.1 8.9 2.5 1.9
 6.1 7.4 2.7 4.8 3.8 6.9 7.8]
Y (targets): [21 47 27 75 30 20 88 60 81 25 85 62 41 42 17 95 30 24 67 69 30 54 35 76
 86]

 Training samples: 16  Testing samples: 9
```


 <img width="230" height="78" alt="image" src="https://github.com/user-attachments/assets/f62ff9ae-c541-4e84-9641-98ce1519b123" />

 
```
Predicted values: [17.04 33.52 74.22 26.73 59.68 39.33 20.92 78.09 69.37]
Actual values   : [20 27 69 30 62 35 24 86 76]
```

<img width="815" height="492" alt="image" src="https://github.com/user-attachments/assets/3950a692-acaa-425a-b3de-94496b8691cb" />



<img width="780" height="487" alt="image" src="https://github.com/user-attachments/assets/5353c255-cfb3-4dd2-acfb-1e5a20bf7cb4" />


```
Mean Absolute Error (MAE): 4.691397441397438
Mean Squared Error (MSE): 25.463280738222547
Root Mean Squared Error (RMSE): 5.046115410711743

Predictions for new hours [2.5 8. ] => [26.73 80.03]
```
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
