from django.shortcuts import render
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

def load_model():
    model = LSTMModel()
    model.load_state_dict(torch.load('crypto_model.pth'))
    model.eval()
    return model

def preprocess_data(data, scaler, prediction_days):
    # Selecionar apenas a coluna 'price' dos dados
    prices = data['price'].values.reshape(-1, 1)
    # Redimensionar e transformar os dados
    scaled_data = scaler.transform(prices)
    # Selecionar os últimos prediction_days dias
    x_test = scaled_data[-prediction_days:]
    # Converter para tensor
    x_test = torch.tensor(x_test, dtype=torch.float32)
    # Retornar os dados preprocessados
    return x_test

def index(request):
    if request.method == 'POST':
        days = int(request.POST.get('days'))
        data = pd.read_csv('bitcoin_prices.csv')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit_transform(data['price'].values.reshape(-1, 1))

        model = load_model()
        x_test = preprocess_data(data, scaler, days)
        with torch.no_grad():
            predicted_price = model(x_test).item()
        predicted_price = scaler.inverse_transform(np.array([[predicted_price]]))

        context = {
            'predicted_price': predicted_price[0][0]
        }
        return render(request, 'prediction/index.html', context)

    return render(request, 'prediction/index.html')
