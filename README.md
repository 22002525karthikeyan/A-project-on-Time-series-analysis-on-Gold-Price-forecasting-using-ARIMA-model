### Name: Karthikeyan R
### Reg.no: 212222240046
### Date:
# EX.NO.09        A project on Time series analysis on Gold Price forecasting using ARIMA model 
### AIM:
To Create a project on Time series analysis on Gold Price forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
data = pd.read_csv('Gold Price.csv')
# Changed the date format to '%Y-%m-%d' to match the actual format in the CSV
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')  
data.set_index('Date', inplace=True)

# Filter data from 2010 onward
data = data[data.index >= '2010-01-01']

# Print the available columns to identify the correct closing price column name
print(data.columns)

# Access the closing price column using the correct name (replace 'Price' with the actual name if different)
# If the column name has extra spaces, use data[' ActualColumnName ']
data['Close'] = pd.to_numeric(data['Price'], errors='coerce')  # Replace 'Price' with the actual column name
data['Close'].fillna(method='ffill', inplace=True)

# Plot the Close price to inspect for trends
plt.figure(figsize=(10, 5))
plt.plot(data['Close'], label='Gold Close Price')
plt.title('Time Series of Gold Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Check stationarity with ADF test
result = adfuller(data['Close'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])


# If p-value > 0.05, apply differencing
data['Close_diff'] = data['Close'].diff().dropna()
result_diff = adfuller(data['Close_diff'].dropna())
print('Differenced ADF Statistic:', result_diff[0])
print('Differenced p-value:', result_diff[1])

# Plot Differenced Representation
plt.figure(figsize=(10, 5))
plt.plot(data['Close_diff'], label='Differenced Close Price', color='red')
plt.title('Differenced Representation of Gold Close Price')
plt.xlabel('Date')
plt.ylabel('Differenced Close Price')
plt.axhline(0, color='black', lw=1, linestyle='--')
plt.legend()
plt.show()

# Use auto_arima to find the optimal (p, d, q) parameters
stepwise_model = auto_arima(data['Close'], start_p=1, start_q=1,
                            max_p=3, max_q=3, seasonal=False, trace=True)
p, d, q = stepwise_model.order
print(stepwise_model.summary())

# Fit the ARIMA model using the optimal parameters
model = sm.tsa.ARIMA(data['Close'], order=(p, d, q))
fitted_model = model.fit()
print(fitted_model.summary())


# Forecast the next 30 days
forecast = fitted_model.forecast(steps=30)
forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')

# Plot actual vs forecasted values
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Actual')
plt.plot(forecast_index, forecast, label='Forecast', color='orange')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('ARIMA Forecast of Gold Price')
plt.legend()
plt.show()

# Evaluate the model with MAE and RMSE
predictions = fitted_model.predict(start=0, end=len(data['Close']) - 1)
mae = mean_absolute_error(data['Close'], predictions)
rmse = np.sqrt(mean_squared_error(data['Close'], predictions))
print('Mean Absolute Error (MAE):', mae)
print('Root Mean Squared Error (RMSE):', rmse)
```
### OUTPUT:
![Untitled](https://github.com/user-attachments/assets/54d16c24-f64b-4d69-adeb-9432eb8e25ac)
![image](https://github.com/user-attachments/assets/616338e2-4e74-46f5-9903-3ae87f0e92d1)
![image](https://github.com/user-attachments/assets/91212584-30ff-451a-a0bc-337a1dd19f63)
![Untitled](https://github.com/user-attachments/assets/8a7f510f-8bbc-4fd8-8955-525cff2af819)
![image](https://github.com/user-attachments/assets/dfe1861c-a68e-4a87-87e1-188c0a88a7e7)
![image](https://github.com/user-attachments/assets/54d4a70b-4c7c-4d6c-ad58-d2fcdabd1bb1)
![Untitled](https://github.com/user-attachments/assets/9d5b3650-8ada-4d05-9224-564c55dcbc6c)
![image](https://github.com/user-attachments/assets/2ddccd27-20e9-4fac-87fc-82a5289c9050)

### RESULT:
Thus the program run successfully based on the ARIMA model using python.
