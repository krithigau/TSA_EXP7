# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 03-05-2025



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```
 import numpy as np
 import pandas as pd
 import matplotlib.pyplot as plt
 from statsmodels.tsa.stattools import adfuller
 from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
 from statsmodels.tsa.ar_model import AutoReg
 from sklearn.metrics import mean_squared_error

data = pd.read_csv('/content/AirPassengers.csv',parse_dates=['Month'],index_col='Month')
result = adfuller(data['#Passengers']) 
print('ADF Statistic:', result[0])
print('p-value:', result[1])

x=int(0.8 * len(data))
train_data = data.iloc[:x]
test_data = data.iloc[x:]

lag_order = 13
model = AutoReg(train_data['#Passengers'], lags=lag_order)
model_fit = model.fit()

plt.figure(figsize=(10, 6))
plot_acf(data['#Passengers'], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.show()
plt.figure(figsize=(10, 6))
plot_pacf(data['#Passengers'], lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

 mse = mean_squared_error(test_data['#Passengers'], predictions)
 print('Mean Squared Error (MSE):', mse)

 plt.figure(figsize=(12, 6))
 plt.plot(test_data['#Passengers'], label='Test Data - Number of passengers')
 plt.plot(predictions, label='Predictions - Number of passengers',linestyle='--')
 plt.xlabel('Date')
 plt.ylabel('Number of passengers')
 plt.title('AR Model Predictions vs Test Data')
 plt.legend()
 plt.grid()
 plt.show()

```
### OUTPUT:

GIVEN DATA

![{5F010481-5E67-40E1-9E0F-D94E01205F1D}](https://github.com/user-attachments/assets/90561668-890e-440e-9651-bd83c8b8f4bb)

PACF - ACF

![{6A493636-C8F2-438C-BA91-95DE2ED7B0F8}](https://github.com/user-attachments/assets/4dff6af2-3782-41bd-a00a-7bca06d9c9e1)

ACCURACY:
![{84DB048C-4B2A-4EC9-A89E-D1737B1C4A3D}](https://github.com/user-attachments/assets/286ffeed-e973-48e8-9345-04ead12d7a68)


FINIAL PREDICTION

![Screenshot 2025-05-06 092951](https://github.com/user-attachments/assets/5691928c-2d21-4cdd-8078-934e028d98f9)

### RESULT:
Thus we have successfully implemented the auto regression function using python.
