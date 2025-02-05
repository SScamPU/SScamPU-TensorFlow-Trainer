import sys
import joblib
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QSpinBox, QTextEdit
)
from PyQt6.QtGui import QTextCursor
from PyQt6.QtCore import pyqtSignal, QThread
from tensorflow.keras.callbacks import EarlyStopping
from model import build_lstm_model
from evaluate import evaluate_model
from data_processing import load_and_preprocess_data  # 我们假设这个函数能够基于 DataFrame 处理数据

class TrainingThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, x_train, y_train, x_test, y_test, epochs, batch_size, scaler):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = scaler  # 训练后需要保存的 scaler

    def run(self):
        self.log_signal.emit("模型训练开始...")
        model = build_lstm_model()

        # 使用 EarlyStopping 防止过拟合
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # 训练模型
        model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test),
                  epochs=self.epochs, batch_size=self.batch_size, callbacks=[early_stopping])

        self.log_signal.emit("模型训练完成！")
        self.finished_signal.emit()

        # 保存模型和标准化器
        model_save_path = 'lstm_model.h5'
        scaler_save_path = 'scaler.pkl'
        model.save(model_save_path)
        joblib.dump(self.scaler, scaler_save_path)
        self.log_signal.emit("模型和标准化器已保存！")

        # 模型评估
        y_pred = model.predict(self.x_test)
        mae, rmse, r2 = evaluate_model(self.y_test, y_pred)
        self.log_signal.emit(f"评估结果 - MAE: {mae}, RMSE: {rmse}, R²: {r2}")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.training_thread = None

    def initUI(self):
        self.setWindowTitle("秋月推理器@AkiTECH 2025")
        self.setGeometry(100, 100, 500, 400)

        layout = QVBoxLayout()

        # 文件选择
        self.file_label = QLabel("未选择文件")
        self.btn_select = QPushButton("选择 CSV 或 XLS 文件")
        self.btn_select.clicked.connect(self.select_file)

        # 训练轮数
        self.epochs_label = QLabel("训练轮数:")
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 500)
        self.epochs_input.setValue(50)

        # 批次大小
        self.batch_label = QLabel("批次大小:")
        self.batch_input = QSpinBox()
        self.batch_input.setRange(1, 128)
        self.batch_input.setValue(32)

        # 训练按钮
        self.btn_train = QPushButton("开始训练")
        self.btn_train.clicked.connect(self.start_training)

        # 滚动日志
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        # 布局
        layout.addWidget(self.file_label)
        layout.addWidget(self.btn_select)
        layout.addWidget(self.epochs_label)
        layout.addWidget(self.epochs_input)
        layout.addWidget(self.batch_label)
        layout.addWidget(self.batch_input)
        layout.addWidget(self.btn_train)
        layout.addWidget(self.log_output)

        self.setLayout(layout)

        self.file_path = ""

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 CSV 或 XLS 文件",
            "",
            "CSV 文件 (*.csv);;Excel 文件 (*.xls *.xlsx)"
        )
        if file_path:
            self.file_path = file_path
            self.file_label.setText(f"📄 {file_path.split('/')[-1]}")

    def start_training(self):
        if not self.file_path:
            self.log_output.append("❌ 请选择数据文件！")
            return
        
        self.log_output.append("📂 加载数据中...")

        # 根据文件扩展名选择合适的加载方式
        try:
            if self.file_path.endswith('.csv'):
                df = pd.read_csv(self.file_path, encoding="utf-8-sig")
            elif self.file_path.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(self.file_path)
            else:
                self.log_output.append("❌ 不支持的文件格式！")
                return
        except Exception as e:
            self.log_output.append(f"❌ 读取文件失败：{e}")
            return

        # 检查必须的列是否存在
        if "timestamp" not in df.columns:
            self.log_output.append("❌ 数据文件中未包含 'timestamp' 列，请检查文件格式！")
            return
        if "voltage" not in df.columns:
            self.log_output.append("❌ 数据文件中未包含 'voltage' 列，请检查文件格式！")
            return

        # 转换 timestamp 列为日期格式
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception as e:
            self.log_output.append(f"❌ 转换 'timestamp' 失败：{e}")
            return

        # 继续数据处理和训练
        # 这里我们调用 load_and_preprocess_data，将 DataFrame 作为参数传入
        try:
            x_train, x_test, y_train, y_test, scaler = load_and_preprocess_data(df)
        except Exception as e:
            self.log_output.append(f"❌ 数据预处理失败：{e}")
            return

        # 获取用户选择的训练轮数和批次大小
        epochs = self.epochs_input.value()
        batch_size = self.batch_input.value()

        self.training_thread = TrainingThread(x_train, y_train, x_test, y_test, epochs, batch_size, scaler)
        self.training_thread.log_signal.connect(self.update_log)
        self.training_thread.finished_signal.connect(self.on_training_finished)
        self.training_thread.start()
        self.update_log("训练线程已启动！")

    def update_log(self, message):
        self.log_output.append(message)
        self.log_output.moveCursor(self.log_output.textCursor().End)

    def on_training_finished(self):
        self.update_log("训练任务已结束！")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
