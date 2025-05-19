"""Các hàm tiện ích"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

class Utils:
    @staticmethod
    def get_data(symbol: str, timeframe: int, bars: int) -> pd.DataFrame:
        """Lấy dữ liệu lịch sử"""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    @staticmethod
    def calculate_trend(timeframe: str) -> str:
        """Tính xu hướng thị trường"""
        df = Utils.get_data("XAUUSD", mt5.TIMEFRAME_D1, 100)
        if df['close'].iloc[-1] > df['close'].iloc[-20]:
            return 'bullish'
        return 'bearish'

    @staticmethod
    def identify_zones(timeframe: str) -> Dict:
        """Xác định các vùng supply/demand"""
        df = Utils.get_data("XAUUSD", mt5.TIMEFRAME_H4, 100)
        
        # Tính toán các mức hỗ trợ/kháng cự
        high = df['high'].max()
        low = df['low'].min()
        
        return {
            'supply': high,
            'demand': low
        }

    @staticmethod
    def check_breakout(price: float, zones: Dict, trend: str) -> bool:
        """Kiểm tra breakout"""
        if trend == 'bullish':
            return price > zones['supply']
        return price < zones['demand']

    @staticmethod
    def log_trade(ticket: int, entry: float, sl: float, tp: float, result: str):
        """Ghi log giao dịch"""
        with open('trade_log.csv', 'a') as f:
            f.write(f"{datetime.now()},{ticket},{entry},{sl},{tp},{result}\n")
