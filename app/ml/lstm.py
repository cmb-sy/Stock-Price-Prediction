import io
import base64
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import pandas as pd

def run_model(from_year, ref_days, code):
    code_dl = code + ".t"

    # ---企業名の取得---
    ticker = yf.Ticker(code_dl)
    if 'longName' not in ticker.info:
        raise ValueError(f"データを取得できませんでした: {code_dl}")
    company_name = ticker.info['longName']

# ---2.データセットの抽出---
    end_date = datetime.now()  # 直近の日付を計算
    start_date = datetime(end_date.year - from_year, 1, 1)  # from_year年前の1月1日を設定
    df = yf.download(code_dl, start=start_date, end=end_date, interval="1d")

    # ---3.データの前処理---
    data = df["Close"]  # Closeコラム（取引終了時の株価）のみ
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset.reshape(-1, 1))

    # ---4.訓練データと検証データの分割---
    training_data_len = int(np.ceil(len(dataset) * 0.7))
    train_data = scaled_data[0:int(training_data_len), :]

    # ---5.訓練データの作成---
    x_train = []
    y_train = []

    for i in tqdm(range(ref_days, len(train_data))):
        x_train.append(train_data[i - ref_days:i, 0])
        y_train.append(train_data[i, 0])
    max_length = max(len(row) for row in x_train)

    x_train_padded = []
    for row in tqdm(x_train):
        if len(row) < max_length:
            row = np.pad(row, (0, max_length - len(row)), 'constant')
        x_train_padded.append(row)

    x_train = np.array(x_train_padded)
    y_train = np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # ---6.モデル構築---
    class LSTMModel(nn.Module):
        def __init__(self):
            super(LSTMModel, self).__init__()
            self.lstm1 = nn.LSTM(input_size=1, hidden_size=128, batch_first=True)
            self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
            self.fc1 = nn.Linear(64, 25)
            self.fc2 = nn.Linear(25, 1)

        def forward(self, x):
            x, _ = self.lstm1(x)
            x, _ = self.lstm2(x)
            x = x[:, -1, :]  # 最後のタイムステップの出力を使用
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # データをTensorに変換
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    # DataLoaderの作成
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # モデルの学習
    model.train()
    for epoch in range(1):
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # ---7.検証データの作成---
    test_data = scaled_data[training_data_len - ref_days:, :]

    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(ref_days, len(test_data)):
        x_test.append(test_data[i - ref_days:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

    # モデルの予測
    model.eval()
    with torch.no_grad():
        predictions = model(x_test_tensor).numpy()

    predictions = scaler.inverse_transform(predictions)

    test_score = np.sqrt(mean_squared_error(y_test, predictions))

    train = data[:training_data_len]
    valid = data[training_data_len:].copy()
    valid.loc[:, 'Predictions'] = predictions

    # ---8.未来の予測---
    future_days = 30  # 予測したい未来の日数
    last_data = scaled_data[-ref_days:]  # 最後のref_days日分のデータを取得

    future_predictions = []
    model.eval()
    with torch.no_grad():
        for _ in range(future_days):
            x_future = torch.tensor(last_data.reshape(1, ref_days, 1), dtype=torch.float32)
            future_pred = model(x_future).numpy()
            future_predictions.append(future_pred[0, 0])
            last_data = np.append(last_data[1:], future_pred)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]

    future_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['Predictions'])

    # ---9.予測結果のプロット---
    plt.switch_backend('Agg')
    plt.figure(figsize=(16, 6))
    plt.title('LSTM Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price of Lasertec', fontsize=18)
    plt.plot(train, label='Train')
    plt.plot(valid, label='Real')
    plt.plot(valid['Predictions'], label='Prediction')
    plt.plot(future_df, label='Future Prediction', linestyle='dashed')
    plt.legend(loc='lower right')

    # 画像を保存
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return plot_url, test_score, company_name