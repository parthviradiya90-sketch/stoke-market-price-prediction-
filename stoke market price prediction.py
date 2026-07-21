import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.metrics import accuracy_score
# Fetching data for Tata Consumer Products (formerly Tata Global Beverages) from Yahoo Finance
# Ticker for Tata Consumer Products on NSE is TATACONSUM.NS
ticker_symbol = 'TATACONSUM.NS'
start_date = '2020-01-01'
end_date = '2024-06-30'

# Download historical data
df_yfinance = yf.download(ticker_symbol, start=start_date, end=end_date)

# Display the first few rows of the yfinance DataFrame
print(f"Data fetched for {ticker_symbol}:")
print(df_yfinance.head(10))

# Now, let's rename the DataFrame to 'date' to match the original notebook's variable name
date = df_yfinance.copy()

date.head(10)
plt.figure(figsize=(16,8))
plt.plot(date.index, date['Close'])
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Tata Global Share Price')
plt.show()

date['open -close'] = date['Open'] - date['Close']
date['high -low'] = date['High'] - date['Low']
date = date.dropna()

x = date[['open -close', 'high -low']]
x.head(10)

y = np.where(date['Close'].shift(-1) > date['Close'], 1, 0)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV

params = {'n_neighbors': np.arange(1, 25)}
knn = neighbors.KNeighborsClassifier()
model = GridSearchCV(knn, params, cv=5)


model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)
print(f"Best parameters: {model.best_params_}")
print(f"Best cross-validation score: {model.best_score_}")
print(f"Test accuracy: {accuracy}")

accuracy_train = accuracy_score(y_train, model.predict(x_train))
print(f"Training accuracy: {accuracy_train}")
accuracy_test = accuracy_score(y_test, model.predict(x_test))
print(f"Test accuracy: {accuracy_test}")

prediction_classification = model.predict(x_test)
actual_prediction = pd.DataFrame({'Actual': y_test.ravel(), 'Predicted': prediction_classification})
actual_prediction.head(10)