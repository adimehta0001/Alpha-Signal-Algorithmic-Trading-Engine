import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np

TICKER = "AAPL" 
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"

def fetch_and_process_data(ticker):
    print(f"--> Fetching deep history for {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print("--> Calculating Advanced Indicators...")
    

    df['RSI'] = df.ta.rsi(length=14)
    
    macd_df = df.ta.macd(fast=12, slow=26, signal=9)
 
    df['MACD'] = macd_df.iloc[:, 0]
    

    bbands = df.ta.bbands(length=20, std=2)

    upper_col = [c for c in bbands.columns if c.startswith('BBU')][0]
    lower_col = [c for c in bbands.columns if c.startswith('BBL')][0]
    
    df['B_UPPER'] = bbands[upper_col]
    df['B_LOWER'] = bbands[lower_col]

    df['SMA_50'] = df.ta.sma(length=50)

    df['Next_Close'] = df['Close'].shift(-1)
    df['Target'] = (df['Next_Close'] > df['Close']).astype(int)
    
    df.dropna(inplace=True)
    
    print(f"--> Success. Dataset size: {len(df)} rows.")
    return df

if __name__ == "__main__":
    data = fetch_and_process_data(TICKER)
    if data is not None:
        data.to_csv(f"{TICKER}_training_data.csv")
        print(f"--> V2 Training Data Saved: {TICKER}_training_data.csv")