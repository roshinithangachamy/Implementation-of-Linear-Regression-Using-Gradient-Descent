# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required library and read the dataframe. 

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate. 

4.Plot the Cost function using Gradient Descent and generate the required graph. 

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: T.Roshini
RegisterNumber:  212223230175

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
  X=np.c_[np.ones(len(X1)), X1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)
    errors=(predictions-y)
    theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
  return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())


X=(data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1: ,-1].values).reshape(-1,1)
print(y)


X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)


theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data=np.array([165349.2,136897,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:

#### Dataset
![image](https://github.com/user-attachments/assets/ad48d1d5-a1a8-41a5-b4f3-c6613ce088e2)

#### X
![image](https://github.com/user-attachments/assets/8fd46ea2-79e3-48a6-9d98-0ef67c7c7e32)

#### Y
![image](https://github.com/user-attachments/assets/b4286698-cbd8-4925-b33b-61192178cb6b)

#### X1_Scaled
![image](https://github.com/user-attachments/assets/6522a8da-3eea-4f15-9961-c65aadcc5853)

#### 
![image](https://github.com/user-attachments/assets/c3b159b8-3ac9-4922-a7b4-8c3742d4c055)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
