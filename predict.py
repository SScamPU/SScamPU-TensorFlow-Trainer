import numpy as np
import joblib
from tensorflow.keras.models import load_model

def make_prediction(model_path, scaler_path, x_test, threshold=120.0):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    y_pred = model.predict(x_test)
    y_pred_real = scaler.inverse_transform(y_pred)

    if y_pred_real[0] > threshold:
        print("⚠️ 超出阈值，需要监控")
    else:
        print("✅ 正常运行")
