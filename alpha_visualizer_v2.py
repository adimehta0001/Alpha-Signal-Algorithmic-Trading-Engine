import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_ta as ta
import numpy as np

TICKER = "AAPL"
print(f"--> Generating V2 Strategy Dashboard for {TICKER}...")

df = yf.download(TICKER, period="6mo")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

bbands = df.ta.bbands(length=20, std=2)
upper_col = [c for c in bbands.columns if c.startswith('BBU')][0]
lower_col = [c for c in bbands.columns if c.startswith('BBL')][0]
df['B_UPPER'] = bbands[upper_col]
df['B_LOWER'] = bbands[lower_col]

macd_df = df.ta.macd(fast=12, slow=26, signal=9)
df['MACD'] = macd_df.iloc[:, 0]
df['MACD_SIGNAL'] = macd_df.iloc[:, 2]

df['RSI'] = df.ta.rsi(length=14)

plt.figure(figsize=(14, 10))
plt.style.use('bmh')

plt.subplot(3, 1, 1)
plt.plot(df['Close'], label='Price', color='black', alpha=0.7)
plt.plot(df['B_UPPER'], label='Upper Band', color='green', linestyle='--', alpha=0.5)
plt.plot(df['B_LOWER'], label='Lower Band', color='red', linestyle='--', alpha=0.5)
plt.fill_between(df.index, df['B_UPPER'], df['B_LOWER'], color='gray', alpha=0.1)
plt.title(f'{TICKER} AI Analysis: Volatility & Trend', fontsize=16)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(df['MACD'], label='MACD Line', color='blue')
plt.plot(df['MACD_SIGNAL'], label='Signal Line', color='orange')
plt.axhline(0, color='black', linewidth=1, linestyle='-')
plt.title('MACD Trend Momentum', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(df['RSI'], label='RSI', color='purple')
plt.axhline(70, color='red', linestyle='--', label='Overbought')
plt.axhline(30, color='green', linestyle='--', label='Oversold')
plt.title('RSI Strength', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{TICKER}_V2_Dashboard.png")
print(f"--> Dashboard saved: {TICKER}_V2_Dashboard.png")
plt.show()