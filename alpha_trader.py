import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

TICKER = "AAPL"
MODEL_FILE = f"{TICKER}_predictor.keras"

print(f"--> Loading V2 Oracle ({MODEL_FILE})...")
model = load_model(MODEL_FILE)

print(f"--> Fetching live market data...")
df = yf.download(TICKER, period="1y") 

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print("--> Calculating indicators...")

df['RSI'] = df.ta.rsi(length=14)

macd_df = df.ta.macd(fast=12, slow=26, signal=9)
df['MACD'] = macd_df.iloc[:, 0]

bbands = df.ta.bbands(length=20, std=2)

upper_col = [c for c in bbands.columns if c.startswith('BBU')][0]
lower_col = [c for c in bbands.columns if c.startswith('BBL')][0]
df['B_UPPER'] = bbands[upper_col]
df['B_LOWER'] = bbands[lower_col]

df['SMA_50'] = df.ta.sma(length=50)

df.dropna(inplace=True)

feature_cols = ['Close', 'RSI', 'MACD', 'B_UPPER', 'B_LOWER', 'SMA_50']
last_60_days = df[feature_cols].tail(60)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(last_60_days)

input_sequence = np.array([scaled_data])

print("--> Asking the AI...")
prediction_prob = model.predict(input_sequence)[0][0]
decision = "BUY" if prediction_prob > 0.5 else "SELL"
confidence = prediction_prob if decision == "BUY" else 1 - prediction_prob

print("\n" + "="*30)
print(f"   ALPHA V2 SIGNAL: {TICKER}")
print("="*30)
print(f"Date:        {df.index[-1].date()}")
print(f"Close Price: ${df['Close'].iloc[-1]:.2f}")
print("-" * 30)
print(f"SIGNAL:      {decision}")
print(f"CONFIDENCE:  {confidence*100:.2f}%")
print("="*30 + "\n")