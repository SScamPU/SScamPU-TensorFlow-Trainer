from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_lstm_model():
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(30, 1)),
        LSTM(16),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
