import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Alpha-Signal Terminal", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #30333d;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Alpha-Signal: Institutional Trading Engine")

st.sidebar.header("Strategy Config")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
time_frame = st.sidebar.selectbox("Data Lookback", ["1y", "2y", "5y"], index=1)

@st.cache_resource
def load_ai_model(ticker):
    try:
        model = load_model(f"{ticker}_predictor.keras")
        return model
    except:
        return None

model = load_ai_model(ticker)

if st.sidebar.button("Initialize System", type="primary"):
    if model is None:
        st.error(f"⚠️ No AI Brain found for {ticker}. Please run brain_engine.py locally first.")
    else:
        with st.spinner(f'📡 Fetching Live Data for {ticker}...'):

            df = yf.download(ticker, period=time_frame)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df['RSI'] = df.ta.rsi(length=14)
            macd_df = df.ta.macd(fast=12, slow=26, signal=9)
            df['MACD'] = macd_df.iloc[:, 0]
            df['MACD_SIGNAL'] = macd_df.iloc[:, 2]
            df['MACD_HIST'] = macd_df.iloc[:, 1]
            
            bbands = df.ta.bbands(length=20, std=2)
            upper_col = [c for c in bbands.columns if c.startswith('BBU')][0]
            lower_col = [c for c in bbands.columns if c.startswith('BBL')][0]
            df['B_UPPER'] = bbands[upper_col]
            df['B_LOWER'] = bbands[lower_col]
            
            df['SMA_50'] = df.ta.sma(length=50)
            df['EMA_200'] = df.ta.ema(length=200)
            df.dropna(inplace=True)

            feature_cols = ['Close', 'RSI', 'MACD', 'B_UPPER', 'B_LOWER', 'SMA_50']
            last_60_days = df[feature_cols].tail(60)
            
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(last_60_days)
            input_sequence = np.array([scaled_data])
            
            prediction_prob = model.predict(input_sequence)[0][0]

            if prediction_prob > 0.60:
                decision = "STRONG BUY"
                signal_color = "green"
            elif prediction_prob < 0.40:
                decision = "STRONG SELL"
                signal_color = "red"
            else:
                decision = "HOLD / NEUTRAL"
                signal_color = "orange"
            
            confidence = prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}", f"{(df['Close'].iloc[-1] - df['Close'].iloc[-2]):.2f}")
            c2.metric("AI Confidence", f"{confidence*100:.1f}%", decision, delta_color="off")
            c3.metric("RSI Momentum", f"{df['RSI'].iloc[-1]:.1f}", "Overbought" if df['RSI'].iloc[-1] > 70 else "Oversold" if df['RSI'].iloc[-1] < 30 else "Neutral")
            c4.metric("Trend (EMA 200)", "BULLISH" if df['Close'].iloc[-1] > df['EMA_200'].iloc[-1] else "BEARISH")

            st.markdown("---")

            st.subheader(f"📊 {ticker} Institutional Technical Analysis")
            
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, 
                                row_heights=[0.6, 0.2, 0.2],
                                subplot_titles=(f"{ticker} Price Action & Volatility", "MACD Momentum", "RSI Strength"))

            fig.add_trace(go.Candlestick(x=df.index,
                            open=df['Open'], high=df['High'],
                            low=df['Low'], close=df['Close'],
                            name='Price'), row=1, col=1)
            

            fig.add_trace(go.Scatter(x=df.index, y=df['B_UPPER'], line=dict(color='rgba(0, 255, 0, 0.5)', width=1), name='Upper Band'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['B_LOWER'], line=dict(color='rgba(255, 0, 0, 0.5)', width=1), name='Lower Band', fill='tonexty', fillcolor='rgba(128, 128, 128, 0.1)'), row=1, col=1)
      
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=2), name='SMA 50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='blue', width=2), name='EMA 200'), row=1, col=1)

            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='cyan', width=2), name='MACD'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_SIGNAL'], line=dict(color='orange', width=2), name='Signal'), row=2, col=1)

            colors = ['green' if val >= 0 else 'red' for val in df['MACD_HIST']]
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_HIST'], marker_color=colors, name='Histogram'), row=2, col=1)

            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=2), name='RSI'), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

            fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("🧠 How the AI made this decision"):
                st.write(f"""
                The Alpha-Signal Engine analyzed **{len(df)} days** of trading data.
                - **The Trend:** Price is currently {'ABOVE' if df['Close'].iloc[-1] > df['EMA_200'].iloc[-1] else 'BELOW'} the 200-Day EMA.
                - **The Volatility:** The Bollinger Bands are {'EXPANDING' if (df['B_UPPER'].iloc[-1] - df['B_LOWER'].iloc[-1]) > (df['B_UPPER'].iloc[-10] - df['B_LOWER'].iloc[-10]) else 'SQUEEZING'}.
                - **The Momentum:** RSI is at {df['RSI'].iloc[-1]:.2f}.
                
                The LSTM Neural Network combined these factors to output a confidence score of **{confidence*100:.2f}%**.
                """)