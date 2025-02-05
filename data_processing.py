import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path, window_size=30):
    df = pd.read_csv(file_path, encoding="utf-8")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    voltage = df["voltage"].values

    scaler = StandardScaler()
    voltage_scaled = scaler.fit_transform(voltage.reshape(-1, 1)).flatten()

    def get_sliding_window(data, window_size):
        return np.array([data[i:i+window_size] for i in range(len(data)-window_size)])

    X = get_sliding_window(voltage_scaled, window_size)
    y = voltage_scaled[window_size:]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    return x_train, x_test, y_train, y_test, scaler
