import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_ta as ta
import numpy as np

TICKER = "AAPL"
print(f"--> Creating the 'Money Shot' chart for {TICKER}...")

df = yf.download(TICKER, period="1y")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

df['SMA_50'] = df.ta.sma(length=50)
df['RSI'] = df.ta.rsi(length=14)


plt.figure(figsize=(14, 8))
plt.style.use('fivethirtyeight') 

plt.subplot(2, 1, 1)
plt.plot(df['Close'], label='Share Price', color='black', alpha=0.6)
plt.plot(df['SMA_50'], label='50-Day SMA (Trend)', color='orange', linestyle='--')
plt.title(f'{TICKER} Alpha-Signal Analysis', fontsize=18)
plt.legend(loc='upper left')

plt.subplot(2, 1, 2)
plt.plot(df['RSI'], label='RSI (Momentum)', color='purple')
plt.axhline(70, color='red', linestyle='--', linewidth=1, label='Overbought (Sell Zone)')
plt.axhline(30, color='green', linestyle='--', linewidth=1, label='Oversold (Buy Zone)')
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig(f"{TICKER}_strategy_chart.png")
print(f"--> Chart saved as {TICKER}_strategy_chart.png. Open it.")
plt.show()