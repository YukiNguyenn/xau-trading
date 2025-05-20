"""
Technical indicators module.

This module provides functions for calculating various technical indicators
used in trading strategies.
"""

import pandas as pd
from pandas_ta import rsi, macd, sma


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        df: DataFrame containing price data
        period: RSI period
        
    Returns:
        Series containing RSI values
    """
    return rsi(df['close'], length=period)


def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        df: DataFrame containing price data
        fast: Fast period
        slow: Slow period
        signal: Signal period
        
    Returns:
        Tuple of (MACD line, Signal line)
    """
    macd_df = macd(df['close'], fast=fast, slow=slow, signal=signal)
    return (
        macd_df[f'MACD_{fast}_{slow}_{signal}'],
        macd_df[f'MACDs_{fast}_{slow}_{signal}']
    )


def calculate_ma(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        df: DataFrame containing price data
        period: MA period
        
    Returns:
        Series containing MA values
    """
    return sma(df['close'], length=period)
