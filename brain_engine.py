import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

TICKER = "AAPL"
FILE_NAME = f"{TICKER}_training_data.csv"
LOOKBACK = 60

print("--> Loading V2 Data...")
df = pd.read_csv(FILE_NAME)

feature_cols = ['Close', 'RSI', 'MACD', 'B_UPPER', 'B_LOWER', 'SMA_50']
target_col = 'Target'

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[feature_cols])
targets = df[target_col].values

X, y = [], []
for i in range(LOOKBACK, len(scaled_features)):
    X.append(scaled_features[i-LOOKBACK:i])
    y.append(targets[i])

X, y = np.array(X), np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


model = Sequential()

model.add(Bidirectional(LSTM(units=100, return_sequences=True), input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.3)) 

model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.3))

model.add(Dense(units=25, activation='relu')) 
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("--> Training V2 Model...")

history = model.fit(X_train, y_train, 
                    epochs=50, 
                    batch_size=32, 
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop])

model.save(f"{TICKER}_predictor.keras")
print(f"--> V2 Model saved. Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")