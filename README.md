# Alpha-Signal: LSTM Quantitative Trading Engine 📈

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-GPU-orange?style=for-the-badge&logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Operational-brightgreen?style=for-the-badge)

**Alpha-Signal** is an algorithmic trading system that leverages **Long Short-Term Memory (LSTM)** neural networks to predict stock market directionality. Unlike traditional linear regression models, this engine utilizes time-series memory to analyze 60-day historical sequences and technical indicators (RSI, SMA, EMA) to generate actionable "Buy" or "Sell" signals.

---

## 🚀 Key Features
* **Automated Data Pipeline:** Fetches real-time OHLCV data using the `yfinance` API.
* **Technical Analysis Engine:** Computes key indicators (RSI, 50-day SMA, 12-day EMA) to feed the neural network.
* **Deep Learning Model:** A 3-layer LSTM architecture trained on 5 years of historical data to detect non-linear market patterns.
* **Visual Strategy Dashboard:** Automatically generates trend analysis charts with buy/sell zones.

## 🛠️ Tech Stack
* **Core:** Python 3.11
* **ML/AI:** TensorFlow (Keras), Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Financial Analysis:** Pandas-TA, YFinance
* **Visualization:** Matplotlib

## 📊 Performance (v1.0)
* **Target Asset:** AAPL (Apple Inc.)
* **Model Accuracy:** ~53-55% (Directional Prediction)
* **Signal Confidence:** Calculated based on sigmoid activation probability.

## 📸 Visualization
*(Add your 'AAPL_strategy_chart.png' here)*

## ⚡ How to Run
1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Train the Model:**
    ```bash
    python brain_engine.py
    ```
3.  **Generate Signals:**
    ```bash
    python alpha_trader.py
    ```

---
*Disclaimer: This project is for educational purposes only. Financial markets are unpredictable, and this model should not be used for real-money trading without further optimization.*    