import requests
import pandas as pd
from datetime import datetime

def get_crypto_data(crypto, vs_currency='usd', days='30'):
    url = f'https://api.coingecko.com/api/v3/coins/{crypto}/market_chart'
    params = {
        'vs_currency': vs_currency,
        'days': days,
        'interval': 'daily'
    }
    response = requests.get(url, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

crypto = 'bitcoin'
data = get_crypto_data(crypto)
data.to_csv(f'{crypto}_prices.csv', index=False)
