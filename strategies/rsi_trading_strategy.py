"""
RSI Trading Strategy

This module implements an RSI-based trading strategy that uses multiple timeframes
and technical indicators to identify potential trading opportunities.
"""

from strategies.core.base_trading_strategy import BaseTradingStrategy
from strategies.core.indicators import calculate_rsi, calculate_macd
from strategies.core.utils import setup_logger
import MetaTrader5 as mt5
import pandas as pd
from pandas_ta import rsi, macd
import time
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional, Any, Union
import logging


class RSITradingStrategy(BaseTradingStrategy):
    """
    RSI-based trading strategy that uses multiple timeframes and technical indicators
    to identify potential trading opportunities.
    
    The strategy combines RSI, MACD, and price action to generate trading signals
    with proper risk management.
    """
    
    def __init__(
        self,
        symbol: str,
        config_path: str,
        mt5_initialized: bool = False,
        logger: Optional[logging.Logger] = None,
        use_mock_data: bool = False,
        initial_balance: Optional[float] = None,
        leverage: int = 2000
    ) -> None:
        """
        Initialize the RSI trading strategy.
        
        Args:
            symbol: Trading symbol
            config_path: Path to configuration file
            mt5_initialized: Whether MT5 is already initialized
            logger: Optional logger for logging strategy events
            use_mock_data: Whether to use mock data (for backtesting)
            initial_balance: Initial balance to use (for backtesting)
            leverage: Trading leverage (default: 2000)
        """
        super().__init__(
            symbol=symbol,
            config_path=config_path,
            mt5_initialized=mt5_initialized,
            logger=logger,
            use_mock_data=use_mock_data,
            initial_balance=initial_balance,
            leverage=leverage
        )
        
        # RSI configuration
        rsi_config = self.config['indicators']['rsi']
        self.rsi_short = rsi_config['periods']['short']
        self.rsi_medium = rsi_config['periods']['medium']
        self.rsi_long = rsi_config['periods']['long']
        self.rsi_short_overbought = rsi_config['thresholds']['short']['overbought']
        self.rsi_short_oversold = rsi_config['thresholds']['short']['oversold']
        self.rsi_medium_overbought = rsi_config['thresholds']['medium']['overbought']
        self.rsi_medium_oversold = rsi_config['thresholds']['medium']['oversold']
        self.rsi_long_overbought = rsi_config['thresholds']['long']['overbought']
        self.rsi_long_oversold = rsi_config['thresholds']['long']['oversold']
        
        # MACD configuration
        macd_config = self.config['indicators']['macd']
        self.macd_fast = macd_config['fast']
        self.macd_slow = macd_config['slow']
        self.macd_signal = macd_config['signal']
        
        # Trading parameters
        self.zone_threshold = self.config['trading']['zone_threshold']
        self.min_volume = self.config['trading']['min_volume']
        self.max_volume = self.config['trading']['max_volume']
        self.spread_points = self.config['costs']['spread']['max_points']
        
        # Initialize trade tracking
        self.open_trades = []

    def calculate_trend(self, timeframe: str) -> str:
        """
        Calculate the market trend based on EMA50 and EMA200.
        
        Args:
            timeframe: Timeframe to analyze ('M15', 'H4', 'D1', etc.)
            
        Returns:
            str: 'bullish', 'bearish', or 'neutral'
        """
        try:
            df = self.get_data(timeframe, 200)
            if df is None or len(df) < 200:
                raise ValueError(f"Not enough data for {timeframe}")
                
            df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
            last_row = df.iloc[-1]
            
            if last_row['ema50'] > last_row['ema200'] * 1.001:  # 0.1% buffer
                return 'bullish'
            elif last_row['ema50'] < last_row['ema200'] * 0.999:  # 0.1% buffer
                return 'bearish'
            return 'neutral'
            
        except Exception as e:
            print(f"Error in calculate_trend: {str(e)}")
            return 'neutral'

    def identify_zones(self, timeframe: str) -> List[Dict[str, Union[str, float, datetime]]]:
        """
        Identify supply and demand zones on the specified timeframe.
        
        Args:
            timeframe: Timeframe to analyze ('M15', 'H4', 'D1', etc.)
            
        Returns:
            List of dictionaries containing zone information
        """
        try:
            df = self.get_data(timeframe, 100)
            if df is None or len(df) < 100:
                raise ValueError(f"Not enough data for {timeframe}")
                
            zones = []
            for i in range(2, len(df) - 2):
                current = df.iloc[i]
                
                # Check for supply zone (resistance)
                if (current['high'] > df.iloc[i-2]['high'] and 
                    current['high'] > df.iloc[i-1]['high'] and 
                    current['high'] > df.iloc[i+1]['high'] and 
                    current['high'] > df.iloc[i+2]['high'] and
                    df.iloc[i+1]['close'] < current['close']):
                    zones.append({
                        'type': 'supply',
                        'price': current['high'],
                        'time': current['time']
                    })
                
                # Check for demand zone (support)
                if (current['low'] < df.iloc[i-2]['low'] and 
                    current['low'] < df.iloc[i-1]['low'] and 
                    current['low'] < df.iloc[i+1]['low'] and 
                    current['low'] < df.iloc[i+2]['low'] and
                    df.iloc[i+1]['close'] > current['close']):
                    zones.append({
                        'type': 'demand',
                        'price': current['low'],
                        'time': current['time']
                    })
                    
            return zones
            
        except Exception as e:
            print(f"Error in identify_zones: {str(e)}")
            return []

    def check_breakout(self, price: float, zones: List[Dict], trend: str) -> bool:
        """
        Check if price has broken out of a supply/demand zone.
        
        Args:
            price: Current price
            zones: List of supply/demand zones
            trend: Current market trend ('bullish' or 'bearish')
            
        Returns:
            bool: True if breakout detected, False otherwise
        """
        for zone in zones[-5:]:  # Only check recent zones
            if zone['type'] == 'supply' and trend == 'bearish':
                if price < zone['price'] * (1 - self.zone_threshold):
                    return True
            elif zone['type'] == 'demand' and trend == 'bullish':
                if price > zone['price'] * (1 + self.zone_threshold):
                    return True
        return False

    def calculate_rsi_indicators(self, df: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate RSI indicators for short, medium, and long periods.
        
        Args:
            df: DataFrame containing price data
            
        Returns:
            Tuple of (rsi_short, rsi_medium, rsi_long) values
        """
        try:
            rsi_short = rsi(df['close'], length=self.rsi_short).iloc[-1]
            rsi_medium = rsi(df['close'], length=self.rsi_medium).iloc[-1]
            rsi_long = rsi(df['close'], length=self.rsi_long).iloc[-1]
            return rsi_short, rsi_medium, rsi_long
        except Exception as e:
            print(f"Error in calculate_rsi_indicators: {str(e)}")
            return None, None, None

    def calculate_macd(self, df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate MACD line and signal line.
        
        Args:
            df: DataFrame containing price data
            
        Returns:
            Tuple of (macd_line, signal_line) values
        """
        try:
            macd_df = macd(
                df['close'], 
                fast=self.macd_fast, 
                slow=self.macd_slow, 
                signal=self.macd_signal
            )
            return macd_df[f'MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'].iloc[-1], \
                   macd_df[f'MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'].iloc[-1]
        except Exception as e:
            print(f"Error in calculate_macd: {str(e)}")
            return None, None

    def check_signals(
        self, 
        m15_df: pd.DataFrame, 
        h4_df: pd.DataFrame, 
        d1_df: pd.DataFrame,
        m5_df: Optional[pd.DataFrame] = None,
        h1_df: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, Optional[str], float]:
        """
        Check for trading signals across multiple timeframes.
        
        Args:
            m15_df: 15-minute timeframe data
            h4_df: 4-hour timeframe data
            d1_df: Daily timeframe data
            m5_df: Optional 5-minute timeframe data
            h1_df: Optional 1-hour timeframe data
            
        Returns:
            Tuple of (signal_found, signal_type, strength) where signal_type is 'buy' or 'sell'
            and strength is a float between 0 and 1 indicating signal strength
        """
        try:
            # Get weekly trend from daily data
            weekly_trend = self.calculate_trend('D1')
            
            # Identify supply/demand zones on H4
            zones = self.identify_zones('H4')
            if not zones:
                return False, None, 0.0
                
            # Get current price from M15
            price = m15_df['close'].iloc[-1]
            
            # Check for breakout from zones
            if not self.check_breakout(price, zones, weekly_trend):
                return False, None, 0.0
            
            # Calculate RSI indicators
            rsi_short, rsi_medium, rsi_long = self.calculate_rsi_indicators(m15_df)
            if None in (rsi_short, rsi_medium, rsi_long):
                return False, None, 0.0
            
            # Calculate MACD
            macd_line, macd_signal = self.calculate_macd(m15_df)
            if None in (macd_line, macd_signal):
                return False, None, 0.0
                
            # Get previous MACD values for crossover detection
            prev_macd, prev_signal = self.calculate_macd(m15_df.iloc[:-1])
            if None in (prev_macd, prev_signal):
                return False, None, 0.0
            
            # Check RSI signals
            rsi_signal, signal_type = self.check_rsi_signals(
                rsi_short, rsi_medium, rsi_long, weekly_trend
            )
            
            if rsi_signal:
                # Check for MACD crossover confirmation
                if (weekly_trend == 'bullish' and signal_type == 'buy' and 
                    prev_macd < prev_signal and macd_line > macd_signal):
                    # Calculate signal strength based on RSI values
                    strength = min(1.0, (self.rsi_short_oversold - rsi_short) / 10 + 
                                 (self.rsi_medium_oversold - rsi_medium) / 20)
                    return True, 'buy', max(0.1, strength)
                elif (weekly_trend == 'bearish' and signal_type == 'sell' and 
                      prev_macd > prev_signal and macd_line < macd_signal):
                    # Calculate signal strength based on RSI values
                    strength = min(1.0, (rsi_short - self.rsi_short_overbought) / 10 + 
                                 (rsi_medium - self.rsi_medium_overbought) / 20)
                    return True, 'sell', max(0.1, strength)
                    
            return False, None, 0.0
            
        except Exception as e:
            print(f"Error in check_signals: {str(e)}")
            return False, None, 0.0

    def check_rsi_signals(
        self, 
        rsi_short: float, 
        rsi_medium: float, 
        rsi_long: float, 
        trend: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check for RSI-based trading signals.
        
        Args:
            rsi_short: Short-term RSI value
            rsi_medium: Medium-term RSI value
            rsi_long: Long-term RSI value
            trend: Current market trend ('bullish' or 'bearish')
            
        Returns:
            Tuple of (signal_found, signal_type) where signal_type is 'buy' or 'sell'
        """
        if None in (rsi_short, rsi_medium, rsi_long):
            return False, None
            
        if trend == 'bullish':
            if (rsi_short < self.rsi_short_oversold and
                rsi_medium < self.rsi_medium_oversold and
                rsi_long < self.rsi_long_oversold):
                return True, 'buy'
        elif trend == 'bearish':
            if (rsi_short > self.rsi_short_overbought and
                rsi_medium > self.rsi_medium_overbought and
                rsi_long > self.rsi_long_overbought):
                return True, 'sell'
                
        return False, None

    def calculate_levels(self, price: float, trend: str) -> Tuple[float, float, float]:
        """
        Calculate entry, stop loss, and take profit levels.
        
        Args:
            price: Current price
            trend: Current market trend ('bullish' or 'bearish')
            
        Returns:
            Tuple of (entry_price, stop_loss, take_profit)
        """
        try:
            # Get ATR for volatility-based position sizing
            atr = self.calculate_atr(self.get_data('M15', 14))
            if atr is None:
                atr = 1.0  # Fallback value
                
            # Get multiplier from config or use default 2.0
            multiplier = self.config['trading'].get('atr_multiplier', 2.0)
            
            if trend == 'bullish':
                entry = price
                sl = price - (atr * multiplier)
                tp = price + (atr * multiplier * 2)  # 2:1 reward:risk ratio
            else:  # bearish
                entry = price
                sl = price + (atr * multiplier)
                tp = price - (atr * multiplier * 2)  # 2:1 reward:risk ratio
                
            return entry, sl, tp
            
        except Exception as e:
            print(f"Error in calculate_levels: {str(e)}")
            # Return safe values if calculation fails
            if trend == 'bullish':
                return price, price * 0.99, price * 1.02  # 1% SL, 2% TP
            else:
                return price, price * 1.01, price * 0.98  # 1% SL, 2% TP

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """
        Calculate Average True Range (ATR) for volatility measurement.
        
        Args:
            df: DataFrame containing price data
            period: Lookback period for ATR calculation
            
        Returns:
            ATR value or None if calculation fails
        """
        try:
            if df is None or len(df) < period + 1:
                return None
                
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR using EMA
            atr = tr.ewm(span=period, adjust=False).mean()
            return atr.iloc[-1]
            
        except Exception as e:
            print(f"Error in calculate_atr: {str(e)}")
            return None

    def calculate_volume(self, risk_amount: float, sl_distance: float) -> float:
        """
        Calculate the trading volume based on risk amount and stop loss distance.
        
        Args:
            risk_amount: Amount to risk per trade (in account currency)
            sl_distance: Distance to stop loss in points
            
        Returns:
            float: Calculated volume that respects min/max volume constraints
        """
        try:
            if sl_distance <= 0:
                self.logger.warning("Invalid stop loss distance, using minimum volume")
                return self.min_volume
                
            # Calculate volume based on risk amount and stop loss distance
            volume = round(risk_amount / (sl_distance * 100), 2)  # Round to 2 decimal places
            
            # Ensure volume is within allowed limits
            volume = max(min(volume, self.max_volume), self.min_volume)
            
            return volume
            
        except Exception as e:
            self.logger.error(f"Error calculating volume: {str(e)}")
            return self.min_volume

    def run_strategy(self) -> None:
        """
        Main strategy execution loop.
        
        Continuously checks for trading signals and executes trades when conditions are met.
        """
        print(f"Starting RSI trading strategy for {self.symbol}")
        
        while True:
            try:
                # Get data from multiple timeframes
                m15_df = self.get_data('M15', 200)
                h4_df = self.get_data('H4', 100)
                d1_df = self.get_data('D1', 200)
                
                if m15_df is None or h4_df is None or d1_df is None:
                    print("Missing data, skipping iteration")
                    time.sleep(60)
                    continue
                
                # Check for trading signals
                signal, signal_type, strength = self.check_signals(m15_df, h4_df, d1_df)
                
                if signal and signal_type in ('buy', 'sell'):
                    price = m15_df['close'].iloc[-1]
                    entry, sl, tp = self.calculate_levels(price, signal_type)
                    
                    # Place the order
                    order_type = (
                        mt5.ORDER_TYPE_BUY if signal_type == 'buy' 
                        else mt5.ORDER_TYPE_SELL
                    )
                    
                    self.place_order(
                        order_type, 
                        self.calculate_volume(self.initial_balance, sl - entry), 
                        entry, 
                        sl, 
                        tp, 
                        "RSITradingStrategy"
                    )
                    print(
                        f"{signal_type.capitalize()} order executed: "
                        f"Entry={entry:.2f}, SL={sl:.2f}, TP={tp:.2f}, Strength={strength:.2f}"
                    )
                
                # Sleep for 15 minutes before next check
                time.sleep(900)
                
            except KeyboardInterrupt:
                print("\nStrategy stopped by user")
                break
                
            except Exception as e:
                print(f"Error in run_strategy: {str(e)}")
                time.sleep(60)  # Wait before retrying on error


def main():
    """
    Main function to initialize and run the trading strategy.
    """
    try:
        strategy = RSITradingStrategy(symbol="XAUUSD")
        strategy.run_strategy()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()