"""
MA (Moving Average) Trading Strategy

This module implements a moving average crossover trading strategy that uses
multiple timeframes to identify trends and generate trading signals.
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any

import MetaTrader5 as mt5
import pandas as pd
from pandas_ta import ema

from strategies.core.base_trading_strategy import BaseTradingStrategy
from strategies.core.indicators import calculate_ma
from strategies.core.utils import setup_logger


class MATradingStrategy(BaseTradingStrategy):
    """
    Moving Average (MA) Crossover Trading Strategy.

    Uses three EMAs (34, 89, 200) to identify trend direction
    and generate trading signals based on crossovers.
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
        Initialize the MA trading strategy.
        
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
        ma_config = self.config["indicators"].get("ma", {})
        self.ma_fast = ma_config.get("fast", 34)
        self.ma_medium = ma_config.get("medium", 89)
        self.ma_slow = ma_config.get("slow", 200)
        self.last_signal_time = None  # Track last signal for deduplication

        if not (self.ma_fast < self.ma_medium < self.ma_slow):
            raise ValueError("MA periods must be in ascending order: fast < medium < slow")
        self.logger.info(f"Initialized MA periods: fast={self.ma_fast}, medium={self.ma_medium}, slow={self.ma_slow}")

    def calculate_trend(self, timeframe: str) -> str:
        """
        Calculate market trend based on EMA crossovers.
        """
        try:
            df = self.get_data(timeframe, max(self.ma_slow * 2, 400))
            if df is None or len(df) < self.ma_slow:
                self.logger.error(f"Not enough data for {timeframe}")
                return "neutral"

            df["ema_fast"] = ema(df["close"], length=self.ma_fast)
            df["ema_medium"] = ema(df["close"], length=self.ma_medium)
            df["ema_slow"] = ema(df["close"], length=self.ma_slow)

            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]

            # Check for bullish trend
            if (last_row["ema_fast"] > last_row["ema_slow"] and 
                prev_row["ema_fast"] <= prev_row["ema_slow"]):
                return "bullish"
            # Check for bearish trend
            if (last_row["ema_fast"] < last_row["ema_slow"] and 
                prev_row["ema_fast"] >= prev_row["ema_slow"]):
                return "bearish"
            return "neutral"

        except Exception as e:
            self.logger.error(f"Error in calculate_trend: {str(e)}")
            return "neutral"

    def calculate_volume(self, price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            price: Current price
            stop_loss: Stop loss price
            
        Returns:
            float: Position size in lots
        """
        try:
            # Get account balance
            balance = self.initial_balance if self.initial_balance is not None else self.balance
            
            # Calculate risk amount (1% of balance)
            risk_amount = balance * 0.01
            
            # Calculate points at risk
            point = mt5.symbol_info(self.symbol).point if mt5.symbol_info(self.symbol) else 0.01
            points_at_risk = abs(price - stop_loss) / point
            
            # Calculate volume based on risk
            volume = risk_amount / (points_at_risk * point)
            
            # Scale volume based on initial balance
            if self.initial_balance is not None:
                # Scale down volume proportionally to initial balance
                volume = volume * (self.initial_balance / 10000)  # Assuming 10000 is standard balance
            
            # Ensure volume is within allowed range
            volume = max(self.min_volume, min(volume, self.max_volume))
            
            # Round to 2 decimal places
            volume = round(volume, 2)
            
            self.logger.debug(
                f"Volume calculation - Balance: {balance:.2f}, "
                f"Risk Amount: {risk_amount:.2f}, "
                f"Points at Risk: {points_at_risk}, "
                f"Calculated Volume: {volume}"
            )
            
            return volume
            
        except Exception as e:
            self.logger.error(f"Error in calculate_volume: {str(e)}")
            return self.min_volume

    def check_signals(
        self, 
        m15_df: pd.DataFrame, 
        h4_df: pd.DataFrame, 
        d1_df: pd.DataFrame,
        m5_df: Optional[pd.DataFrame] = None,
        h1_df: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, Optional[str], float]:
        """Check for trading signals based on EMA crossovers."""
        try:
            if m15_df is None or len(m15_df) < self.ma_slow + 1:
                self.logger.warning("Insufficient M15 data for EMA calculation")
                return False, None, 0.0

            # Calculate EMAs
            ema_fast = ema(m15_df["close"], length=self.ma_fast)
            ema_medium = ema(m15_df["close"], length=self.ma_medium)
            ema_slow = ema(m15_df["close"], length=self.ma_slow)

            if ema_fast is None or ema_medium is None or ema_slow is None:
                self.logger.error("Failed to calculate EMAs")
                return False, None, 0.0

            current_fast = ema_fast.iloc[-1]
            current_medium = ema_medium.iloc[-1]
            current_slow = ema_slow.iloc[-1]
            prev_fast = ema_fast.iloc[-2]
            prev_medium = ema_medium.iloc[-2]
            prev_slow = ema_slow.iloc[-2]
            current_time = m15_df["time"].iloc[-1]

            # Calculate signal strength based on EMA distances
            fast_medium_distance = abs(current_fast - current_medium) / m15_df["close"].iloc[-1]
            fast_slow_distance = abs(current_fast - current_slow) / m15_df["close"].iloc[-1]
            signal_strength = min(1.0, (fast_medium_distance + fast_slow_distance) * 50)  # Scale to 0-1 range
            
            self.logger.debug(
                f"EMA Signals - Fast: {current_fast:.2f}, Medium: {current_medium:.2f}, "
                f"Slow: {current_slow:.2f}, Strength: {signal_strength:.2f}"
            )

            # Prevent duplicate signals
            if self.last_signal_time == current_time:
                self.logger.debug(f"Skipping duplicate signal at {current_time}")
                return False, None, 0.0

            # Check for buy signals
            if (prev_fast <= prev_slow and current_fast > current_slow):  # Main buy signal
                self.last_signal_time = current_time
                self.logger.info("Buy signal: Fast EMA crossed above Slow EMA")
                return True, "buy", signal_strength
            if (prev_fast <= prev_medium and current_fast > current_medium):  # Secondary buy signal
                self.last_signal_time = current_time
                self.logger.info("Buy signal: Fast EMA crossed above Medium EMA")
                return True, "buy", signal_strength * 0.8  # Lower strength for secondary signal

            # Check for sell signals
            if (prev_fast >= prev_slow and current_fast < current_slow):  # Main sell signal
                self.last_signal_time = current_time
                self.logger.info("Sell signal: Fast EMA crossed below Slow EMA")
                return True, "sell", signal_strength
            if (prev_fast >= prev_medium and current_fast < current_medium):  # Secondary sell signal
                self.last_signal_time = current_time
                self.logger.info("Sell signal: Fast EMA crossed below Medium EMA")
                return True, "sell", signal_strength * 0.8  # Lower strength for secondary signal

            return False, None, 0.0

        except Exception as e:
            self.logger.error(f"Error in check_signals: {str(e)}")
            return False, None, 0.0

    def calculate_levels(self, price: float, signal_type: str) -> Tuple[float, float, float]:
        """
        Calculate entry, stop loss, and take profit levels based on ATR.
        """
        try:
            atr = self.calculate_atr(self.get_data("M15", 100))  # Reduced bars for speed
            if atr is None or atr <= 0:
                self.logger.warning("Invalid ATR, using default")
                atr = 1.0
            multiplier = self.config["trading"].get("atr_multiplier", 2.0)

            point = mt5.symbol_info(self.symbol).point if mt5.symbol_info(self.symbol) else 0.01
            if signal_type == "buy":
                entry = price
                sl = entry - (atr * multiplier)
                tp = entry + (atr * multiplier * 2)
            else:  # sell
                entry = price
                sl = entry + (atr * multiplier)
                tp = entry - (atr * multiplier * 2)

            precision = self.config["trading"].get("price_precision", 2)
            sl = round(sl, precision)
            tp = round(tp, precision)
            self.logger.debug(f"Levels for {signal_type}: Entry={entry:.2f}, SL={sl:.2f}, TP={tp:.2f}")
            return entry, sl, tp

        except Exception as e:
            self.logger.error(f"Error in calculate_levels: {str(e)}")
            if signal_type == "buy":
                return price, price * 0.99, price * 1.02
            return price, price * 1.01, price * 0.98

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """
        Calculate Average True Range (ATR) for volatility measurement.
        """
        try:
            if df is None or len(df) < period + 1:
                self.logger.warning("Insufficient data for ATR calculation")
                return None

            high = df["high"]
            low = df["low"]
            close = df["close"]

            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            atr = tr.ewm(span=period, adjust=False).mean()
            atr_value = atr.iloc[-1]
            if atr_value <= 0:
                self.logger.warning("ATR is zero or negative")
                return None
            return atr_value

        except Exception as e:
            self.logger.error(f"Error in calculate_atr: {str(e)}")
            return None

    def run_strategy(self) -> None:
        """
        Main strategy execution loop.
        """
        self.logger.info(f"Starting MA trading strategy for {self.symbol}")

        while True:
            try:
                # Get more M15 data to ensure enough bars for EMA calculation
                m15_df = self.get_data("M15", max(self.ma_slow * 2, 400))
                h4_df = self.get_data("H4", 100)
                d1_df = self.get_data("D1", 200)

                if m15_df is None or h4_df is None or d1_df is None:
                    self.logger.warning("Missing data, skipping iteration")
                    time.sleep(60)
                    continue

                if len(m15_df) < self.ma_slow + 1:
                    self.logger.warning(f"Insufficient M15 data: {len(m15_df)} bars, need {self.ma_slow + 1}")
                    time.sleep(60)
                    continue

                signal, signal_type, strength = self.check_signals(m15_df, h4_df, d1_df)

                if signal and signal_type in ("buy", "sell"):
                    tick = mt5.symbol_info_tick(self.symbol)
                    if not tick:
                        self.logger.error("Failed to get tick data")
                        time.sleep(60)
                        continue
                    price = tick.ask if signal_type == "buy" else tick.bid
                    entry, sl, tp = self.calculate_levels(price, signal_type)

                    order_type = mt5.ORDER_TYPE_BUY if signal_type == "buy" else mt5.ORDER_TYPE_SELL

                    self.place_order(
                        order_type=order_type,
                        volume=self.calculate_volume(price, sl),
                        price=entry,
                        sl=sl,
                        tp=tp,
                        strategy_name="MATradingStrategy",
                        rsi_values=(None, None, None),
                        macd_crossover=False,
                        breakout_distance=strength,
                    )
                    self.logger.info(
                        f"{signal_type.capitalize()} order executed: "
                        f"Entry={entry:.2f}, SL={sl:.2f}, TP={tp:.2f}, Strength={strength:.2f}"
                    )
                    time.sleep(900)  # Wait 15 minutes after order

                else:
                    time.sleep(300)  # Wait 5 minutes if no signal

            except KeyboardInterrupt:
                self.logger.info("Strategy stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in run_strategy: {str(e)}")
                time.sleep(60)

def main() -> None:
    """Test the strategy."""
    logging.basicConfig(level=logging.DEBUG)
    if not mt5.initialize():
        logging.error("Failed to initialize MT5")
        return
    try:
        strategy = MATradingStrategy(symbol="XAUUSD")
        m15_df = strategy.get_data("M15", 100)
        h4_df = strategy.get_data("H4", 100)
        d1_df = strategy.get_data("D1", 100)
        if all(df is not None for df in [m15_df, h4_df, d1_df]):
            signal, signal_type, strength = strategy.check_signals(m15_df, h4_df, d1_df)
            logging.info(f"Signal: {signal}, Type: {signal_type}, Strength: {strength}")
        else:
            logging.error("Failed to get data for testing")
    except Exception as e:
        logging.error(f"Error: {str(e)}")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()