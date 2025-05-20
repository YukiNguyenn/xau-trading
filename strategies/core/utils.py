"""
Utility functions module.

This module provides various utility functions used across the trading system.
"""

import logging
import MetaTrader5 as mt5
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logger with specified name and level.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger


class Utils:
    """
    Utility functions for trading system.
    """
    
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
        """
        Identify trading zones based on timeframe.
        
        Args:
            timeframe: Trading timeframe
            
        Returns:
            Dict: Dictionary containing zone information
        """
        zones = {
            'M15': {
                'support': [],
                'resistance': [],
                'trend': 'neutral'
            },
            'H4': {
                'support': [],
                'resistance': [],
                'trend': 'neutral'
            },
            'D1': {
                'support': [],
                'resistance': [],
                'trend': 'neutral'
            }
        }
        return zones.get(timeframe, {})

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

    @staticmethod
    def calculate_breakout_distance(
        price: float,
        level: float,
        direction: str
    ) -> float:
        """
        Calculate distance from breakout level.
        
        Args:
            price: Current price
            level: Breakout level
            direction: Breakout direction ('up' or 'down')
            
        Returns:
            float: Distance from breakout level
        """
        if direction == 'up':
            return (price - level) / level * 100
        else:
            return (level - price) / level * 100
            
    @staticmethod
    def format_timeframe(timeframe: str) -> int:
        """
        Convert timeframe string to MT5 timeframe value.
        
        Args:
            timeframe: Timeframe string (e.g., 'M15', 'H4', 'D1')
            
        Returns:
            int: MT5 timeframe value
        """
        timeframes = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        return timeframes.get(timeframe, mt5.TIMEFRAME_M15)
