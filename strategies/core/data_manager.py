"""
Data management module.

This module provides functionality for managing market data, including
data fetching, preprocessing, and technical analysis.
"""

import logging
import MetaTrader5 as mt5
import pandas as pd
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta


class DataManager:
    """
    Manages market data and technical analysis.
    """
    
    def __init__(
        self,
        symbol: str,
        timeframes: Dict[str, int],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the data manager.
        
        Args:
            symbol: Trading symbol
            timeframes: Dictionary of timeframe names and MT5 timeframe values
            logger: Optional logger instance
        """
        self.symbol = symbol
        self.timeframes = timeframes
        self.logger = logger or logging.getLogger(__name__)
        
    def fetch_data(
        self,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        num_bars: int = 1000
    ) -> Optional[pd.DataFrame]:
        """
        Fetch market data for the specified timeframe.
        
        Args:
            timeframe: Timeframe name (e.g., 'M15', 'H4', 'D1')
            start_date: Start date for data
            end_date: End date for data
            num_bars: Number of bars to fetch if dates not specified
            
        Returns:
            Optional[pd.DataFrame]: DataFrame containing market data
        """
        try:
            if timeframe not in self.timeframes:
                self.logger.error(f"Invalid timeframe: {timeframe}")
                return None
                
            if start_date is None:
                start_date = datetime.now() - timedelta(days=30)
            if end_date is None:
                end_date = datetime.now()
                
            rates = mt5.copy_rates_range(
                self.symbol,
                self.timeframes[timeframe],
                start_date,
                end_date
            )
            
            if rates is None or len(rates) == 0:
                self.logger.error(f"No data received for {timeframe}")
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            self.logger.info(
                f"Fetched {len(df)} bars for {timeframe} "
                f"from {df.index[0]} to {df.index[-1]}"
            )
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            return None
            
    def calculate_atr(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            df: DataFrame containing price data
            period: ATR period
            
        Returns:
            pd.Series: ATR values
        """
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series()
            
    def calculate_support_resistance(
        self,
        df: pd.DataFrame,
        window: int = 20,
        threshold: float = 0.002
    ) -> Tuple[List[float], List[float]]:
        """
        Calculate support and resistance levels.
        
        Args:
            df: DataFrame containing price data
            window: Window size for finding local extrema
            threshold: Minimum distance between levels
            
        Returns:
            Tuple[List[float], List[float]]: Support and resistance levels
        """
        try:
            highs = df['high'].rolling(window=window, center=True).max()
            lows = df['low'].rolling(window=window, center=True).min()
            
            resistance_levels = []
            support_levels = []
            
            # Find resistance levels
            for i in range(window, len(df) - window):
                if highs.iloc[i] == df['high'].iloc[i]:
                    level = df['high'].iloc[i]
                    if not any(abs(r - level) < threshold for r in resistance_levels):
                        resistance_levels.append(level)
                        
            # Find support levels
            for i in range(window, len(df) - window):
                if lows.iloc[i] == df['low'].iloc[i]:
                    level = df['low'].iloc[i]
                    if not any(abs(s - level) < threshold for s in support_levels):
                        support_levels.append(level)
                        
            return sorted(support_levels), sorted(resistance_levels)
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {str(e)}")
            return [], []
            
    def detect_trend(
        self,
        df: pd.DataFrame,
        ma_period: int = 20
    ) -> str:
        """
        Detect market trend using moving average.
        
        Args:
            df: DataFrame containing price data
            ma_period: Moving average period
            
        Returns:
            str: 'bullish', 'bearish', or 'neutral'
        """
        try:
            ma = df['close'].rolling(window=ma_period).mean()
            current_price = df['close'].iloc[-1]
            current_ma = ma.iloc[-1]
            
            if current_price > current_ma * 1.002:
                return 'bullish'
            elif current_price < current_ma * 0.998:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            self.logger.error(f"Error detecting trend: {str(e)}")
            return 'neutral' 