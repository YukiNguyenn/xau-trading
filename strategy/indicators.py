"""Các chỉ báo kỹ thuật cho chiến lược giao dịch"""

import numpy as np
import pandas as pd
from pandas_ta import rsi, macd

class TechnicalIndicators:
    def __init__(self, rsi_params, macd_params):
        self.rsi_params = rsi_params
        self.macd_params = macd_params

    def calculate_rsi_indicators(self, df):
        """Tính các chỉ số RSI"""
        rsi_short = rsi(df['close'], length=self.rsi_params['short_period'])
        rsi_medium = rsi(df['close'], length=self.rsi_params['medium_period'])
        rsi_long = rsi(df['close'], length=self.rsi_params['long_period'])
        
        return rsi_short.iloc[-1], rsi_medium.iloc[-1], rsi_long.iloc[-1]

    def calculate_macd(self, df, fast=None, slow=None, signal=None):
        """Tính MACD"""
        if fast is None:
            fast = self.macd_params['fast']
        if slow is None:
            slow = self.macd_params['slow']
        if signal is None:
            signal = self.macd_params['signal']
            
        macd_result = macd(df['close'], fast=fast, slow=slow, signal=signal)
        return macd_result.iloc[-1]['MACD_12_26_9'], macd_result.iloc[-1]['MACDs_12_26_9']

    def check_rsi_signals(self, rsi_short, rsi_medium, rsi_long, trend):
        """Kiểm tra tín hiệu RSI"""
        if trend == 'bullish':
            if rsi_short > self.rsi_params['short_overbought']:
                return False, "RSI Short Overbought"
            if rsi_medium > self.rsi_params['medium_overbought']:
                return False, "RSI Medium Overbought"
            if rsi_long > self.rsi_params['long_overbought']:
                return False, "RSI Long Overbought"
            return True, "RSI Signal Confirmed"
        else:  # bearish
            if rsi_short < self.rsi_params['short_oversold']:
                return False, "RSI Short Oversold"
            if rsi_medium < self.rsi_params['medium_oversold']:
                return False, "RSI Medium Oversold"
            if rsi_long < self.rsi_params['long_oversold']:
                return False, "RSI Long Oversold"
            return True, "RSI Signal Confirmed"
