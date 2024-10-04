import ccxt
import pandas as pd
import numpy as np
# Initialize the exchange
exchange = ccxt.binance()

# Fetch historical candlestick data (5 channels: Open, High, Low, Close, Volume)
# Symbol: 'TURBO/USDT' or the pair you are interested in
# Timeframe: '1m', '5m', '1h', '1d' (adjust as needed)
# Limit: Number of data points (candlesticks)
timeFrameFactorTable = {
    '1h' : 10000,
    '15m': 100000
}
    

exchange = ccxt.binance({
    'options': {
        'defaultType': 'future',  # Specify futures market for swaps
    }
})

symbol = 'TURBO/USDT'
timeframe = '15m'  # One-minute candlesticks
limit = 1000  # Number of candlesticks to fetch

candles = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

# Convert to a pandas DataFrame for easy viewing and manipulation
df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

for col in ['open', 'high', 'low', 'close']:
    df[col] *= timeFrameFactorTable[timeframe]



# Calculate (close - open) ^ 2
df['diff_squared'] = (df['close'] - df['open']) ** 2

# Apply the sign based on whether the candlestick is positive or negative
df['signed_diff_squared'] = df.apply(lambda row: row['diff_squared'] if row['close'] > row['open'] else -row['diff_squared'], axis=1)

# Calculate the sum of the future 5 candlesticks
df['sum_future_5'] = df['signed_diff_squared'].rolling(window=5).sum().shift(-4)

df.dropna(inplace=True)
# Keep only the 'sum_future_5' column

df = df[[ 'open', 'high', 'low', 'close', 'volume', 'sum_future_5']]


data = df.to_numpy()

# Step 2: Separate the features (X) and the target (y)
x = data[:, :-1]  # All columns except the last one (features: open, high, low, close, volume)
y = data[:, -1]   # The last column is 'sum_future_5'

timeStampsPerSample = 50
stride = 10

X_time_series = []
y_time_series = []

for i in range(0,len(x) - timeStampsPerSample + 1, stride):
    X_time_series.append(x[i:i + timeStampsPerSample])  # Append n consecutive samples for input
    y_time_series.append(y[i + timeStampsPerSample - 1])  # Use the nth sample's sum_future_5 as the target

# Convert lists to numpy arrays
x = np.array(X_time_series)  # Shape: (num_samples, n, 5)
x = np.transpose(x, (0, 2, 1))
y = np.array(y_time_series)  # Shape: (num_samples,)



# Step 3: Check the shapes
print("X shape (features):", x.shape)  # Should be (n_samples, 5) for 5 channels
print("y shape (target):", y.shape)    # Should be (n_samples,)

print(y[-100:])
# Display the DataFrame with the new column
