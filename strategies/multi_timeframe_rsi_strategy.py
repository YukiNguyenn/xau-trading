import logging
import os
import time
from typing import Optional, Tuple

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from pandas_ta import rsi

from strategies.core.base_trading_strategy import BaseTradingStrategy
from strategies.core.indicators import calculate_rsi, calculate_macd
from strategies.core.utils import setup_logger


class MultiTimeframeRSIStrategy(BaseTradingStrategy):
    """
    Multi-Timeframe RSI trading strategy using H1, M15, and M5 RSI levels.

    Sell conditions (at least two timeframes overbought, including H1):
    - H1 RSI > overbought threshold (from config)
    - M15 or M5 RSI > overbought threshold

    Buy conditions (at least two timeframes oversold, including H1):
    - H1 RSI < oversold threshold
    - M15 or M5 RSI < oversold threshold
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
        Initialize the Multi-Timeframe RSI trading strategy.
        
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
        self.last_signal_time = None  # Track last signal for deduplication

        # RSI configuration from config.json
        rsi_config = self.config["indicators"]["rsi"]
        self.timeframe_levels = {
            "H1": rsi_config["thresholds"]["long"],
            "M15": rsi_config["thresholds"]["medium"],
            "M5": rsi_config["thresholds"]["short"],
        }
        self.rsi_periods = {
            "H1": rsi_config["periods"]["long"],
            "M15": rsi_config["periods"]["medium"],
            "M5": rsi_config["periods"]["short"],
        }
        # Thêm tham số cho ATR và rủi ro
        self.atr_multiplier = self.config.get("trading", {}).get("atr_multiplier", 2.0)
        self.risk_percentage = self.config.get("trading", {}).get("risk_percentage", 0.01)

    def _setup_logger(self, name: str) -> logging.Logger:
        """Set up a logger for the strategy."""
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        log_file = os.path.join(log_dir, f"{name.lower()}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def _generate_mock_data(self, timeframe: str, bars: int) -> pd.DataFrame:
        """Generate mock price data for testing when MT5 data is unavailable."""
        delta = {
            "M5": pd.Timedelta(minutes=5),
            "M15": pd.Timedelta(minutes=15),
            "H1": pd.Timedelta(hours=1),
            "D1": pd.Timedelta(days=1),
        }.get(timeframe, pd.Timedelta(minutes=15))
        end_date = pd.Timestamp.now()
        times = pd.date_range(end=end_date, periods=bars, freq=delta)
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, bars - 1)  # Increased volatility
        prices = np.cumprod(1 + returns) * 2000
        prices = np.insert(prices, 0, 2000)
        df = pd.DataFrame({
            "time": times,
            "open": prices * 0.999 + np.random.normal(0, 0.2, bars),
            "high": prices * 1.001 + np.abs(np.random.normal(0, 0.2, bars)),
            "low": prices * 0.998 - np.abs(np.random.normal(0, 0.2, bars)),
            "close": prices,
            "tick_volume": np.random.randint(100, 1000, bars),
            "spread": np.random.randint(1, 5, bars),
            "real_volume": np.random.randint(100, 1000, bars),
        })
        self.logger.info(f"Generated {len(df)} mock {timeframe} bars for {self.symbol}")
        return df

    def get_data(self, timeframe: str, bars: int, retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch historical data from MT5 with retries."""
        if timeframe not in self.timeframes:
            self.logger.error(f"Invalid timeframe: {timeframe}")
            return self._generate_mock_data(timeframe, bars)
        for attempt in range(retries):
            try:
                rates = mt5.copy_rates_from_pos(self.symbol, self.timeframes[timeframe], 0, bars)
                if rates is None or len(rates) == 0:
                    self.logger.warning(f"No data for {self.symbol} {timeframe} on attempt {attempt + 1}")
                    if attempt == retries - 1:
                        return self._generate_mock_data(timeframe, bars)
                    time.sleep(1)
                    continue
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                self.logger.debug(f"Retrieved {len(df)} {timeframe} bars for {self.symbol}")
                return df
            except Exception as e:
                self.logger.error(f"Error retrieving {timeframe} data on attempt {attempt + 1}: {str(e)}")
                if attempt == retries - 1:
                    return self._generate_mock_data(timeframe, bars)
                time.sleep(1)
        return None

    def calculate_rsi(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """Calculate RSI for the given dataframe."""
        try:
            if df is None or len(df) < period:
                self.logger.warning(f"Insufficient data for RSI calculation: {len(df)} bars, need {period}")
                return pd.Series()
            return rsi(df["close"], length=period)
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series()

    def get_rsi_signal(self, timeframe: str, df: Optional[pd.DataFrame] = None) -> Tuple[float, str]:
        """Get RSI signal for a specific timeframe."""
        try:
            if df is None:
                df = self.get_data(timeframe, 200)
            if df is None or len(df) < 2:
                self.logger.warning(f"Insufficient data for {timeframe} RSI")
                return 0.0, "neutral"
            period = self.rsi_periods.get(timeframe, self.rsi_periods["M15"])
            rsi_values = self.calculate_rsi(df, period)
            if rsi_values.empty or rsi_values.iloc[-1] is None or np.isnan(rsi_values.iloc[-1]):
                return 0.0, "neutral"
            current_rsi = rsi_values.iloc[-1]
            levels = self.timeframe_levels.get(timeframe, {})
            overbought = levels.get("overbought", 70)
            oversold = levels.get("oversold", 30)
            if current_rsi >= overbought:
                return current_rsi, "overbought"
            if current_rsi <= oversold:
                return current_rsi, "oversold"
            return current_rsi, "neutral"
        except Exception as e:
            self.logger.error(f"Error in get_rsi_signal for {timeframe}: {str(e)}")
            return 0.0, "neutral"

    def determine_trend(self, d1_df: pd.DataFrame) -> str:
        """Determine market trend using EMA200 on D1."""
        if d1_df is None or len(d1_df) < 200:
            self.logger.warning("Insufficient D1 data for trend determination")
            return "neutral"
        ema200 = d1_df['close'].ewm(span=200, adjust=False).mean().iloc[-1]
        last_close = d1_df['close'].iloc[-1]
        if last_close > ema200:
            return "bullish"
        elif last_close < ema200:
            return "bearish"
        else:
            return "neutral"

    def is_bullish_crossover(self, macd: pd.Series, signal: pd.Series, lookback: int = 3) -> bool:
        """Check for bullish MACD crossover in the last 'lookback' bars."""
        for i in range(1, lookback + 1):
            if len(macd) < i + 1:
                return False
            if macd.iloc[-i - 1] < signal.iloc[-i - 1] and macd.iloc[-i] > signal.iloc[-i]:
                return True
        return False

    def is_bearish_crossover(self, macd: pd.Series, signal: pd.Series, lookback: int = 3) -> bool:
        """Check for bearish MACD crossover in the last 'lookback' bars."""
        for i in range(1, lookback + 1):
            if len(macd) < i + 1:
                return False
            if macd.iloc[-i - 1] > signal.iloc[-i - 1] and macd.iloc[-i] < signal.iloc[-i]:
                return True
        return False

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR (Average True Range) on the dataframe."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        return atr

    def check_signals(
        self,
        m15_df: pd.DataFrame,
        h4_df: pd.DataFrame,
        d1_df: pd.DataFrame,
        m5_df: Optional[pd.DataFrame] = None,
        h1_df: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, Optional[str], float]:
        """Check for trading signals based on RSI, trend, and MACD confirmation."""
        try:
            if m15_df is None or len(m15_df) < self.rsi_periods["M15"]:
                self.logger.warning("Insufficient M15 data")
                return False, None, 0.0
            current_time = m15_df["time"].iloc[-1]
            if self.last_signal_time == current_time:
                self.logger.debug(f"Skipping duplicate signal at {current_time}")
                return False, None, 0.0
            h1_rsi, h1_signal = self.get_rsi_signal("H1", h1_df)
            m15_rsi, m15_signal = self.get_rsi_signal("M15", m15_df)
            m5_rsi, m5_signal = self.get_rsi_signal("M5", m5_df)
            self.logger.debug(
                f"RSI Signals - H1: {h1_rsi:.2f} ({h1_signal}), "
                f"M15: {m15_rsi:.2f} ({m15_signal}), M5: {m5_rsi:.2f} ({m5_signal})"
            )
            strength = 0.0
            potential_signal = None

            if h1_signal == "overbought" and (m15_signal == "overbought" or m5_signal == "overbought"):
                potential_signal = "sell"
                strength = min(1.0, (h1_rsi - self.timeframe_levels["H1"]["overbought"]) / 10 + 
                                    (m5_rsi - self.timeframe_levels["M5"]["overbought"]) / 20)
            elif h1_signal == "oversold" and (m15_signal == "oversold" or m5_signal == "oversold"):
                potential_signal = "buy"
                strength = min(1.0, (self.timeframe_levels["H1"]["oversold"] - h1_rsi) / 5 + 
                                    (self.timeframe_levels["M5"]["oversold"] - m5_rsi) / 10)

            if potential_signal is None:
                return False, None, 0.0

            # Check D1 trend alignment
            trend = self.determine_trend(d1_df)
            if (potential_signal == "buy" and trend != "bullish") or \
               (potential_signal == "sell" and trend != "bearish"):
                self.logger.info(f"Signal {potential_signal} does not align with trend {trend}")
                return False, None, 0.0

            # MACD confirmation on M15
            macd, signal_line = calculate_macd(m15_df['close'], fast=12, slow=26, signal=9)
            if potential_signal == "buy" and not self.is_bullish_crossover(macd, signal_line):
                self.logger.info("No bullish MACD crossover for buy signal")
                return False, None, 0.0
            elif potential_signal == "sell" and not self.is_bearish_crossover(macd, signal_line):
                self.logger.info("No bearish MACD crossover for sell signal")
                return False, None, 0.0

            self.last_signal_time = current_time
            self.logger.info(f"{potential_signal.capitalize()} signal confirmed with trend and MACD")
            return True, potential_signal, max(0.1, strength)
        except Exception as e:
            self.logger.error(f"Error in check_signals: {str(e)}")
            return False, None, 0.0

    def calculate_levels(self, price: float, trend: str, df: pd.DataFrame = None) -> Tuple[float, float, float]:
        """Calculate entry, stop loss, and take profit levels using ATR."""
        try:
            if df is not None:
                atr = self.calculate_atr(df)
                if trend == "buy":
                    entry = price
                    stop_loss = entry - self.atr_multiplier * atr
                    take_profit = entry + self.atr_multiplier * 2 * atr  # 2:1 ratio
                else:
                    entry = price
                    stop_loss = entry + self.atr_multiplier * atr
                    take_profit = entry - self.atr_multiplier * 2 * atr
            else:
                point = mt5.symbol_info(self.symbol).point if mt5.symbol_info(self.symbol) else 0.01
                if trend == "buy":
                    entry = price
                    stop_loss = entry - self.stop_loss_points * point
                    take_profit = entry + self.take_profit_points * point
                else:
                    entry = price
                    stop_loss = entry + self.stop_loss_points * point
                    take_profit = entry - self.take_profit_points * point
            return entry, stop_loss, take_profit
        except Exception as e:
            self.logger.error(f"Error in calculate_levels: {str(e)}")
            return price, price * 0.99, price * 1.01

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
        """Execute the trading strategy with dynamic risk management."""
        try:
            m5_df = self.get_data("M5", 400)
            m15_df = self.get_data("M15", 200)
            h1_df = self.get_data("H1", 200)
            h4_df = self.get_data("H4", 100)
            d1_df = self.get_data("D1", 100)
            if any(df is None for df in [m5_df, m15_df, h1_df, h4_df, d1_df]):
                self.logger.error("Failed to get required data")
                time.sleep(60)
                return
            signal, signal_type, strength = self.check_signals(m15_df, h4_df, d1_df, m5_df, h1_df)
            if not signal or signal_type not in ["buy", "sell"]:
                time.sleep(300)
                return
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                self.logger.error("Failed to get tick data")
                time.sleep(60)
                return
            current_price = tick.ask if signal_type == "buy" else tick.bid
            entry, sl, tp = self.calculate_levels(current_price, signal_type, m15_df)

            # Calculate volume based on risk
            if self.use_mock_data:
                balance = self.initial_balance
            else:
                account_info = mt5.account_info()
                if account_info is None:
                    self.logger.error("Failed to get account info")
                    time.sleep(60)
                    return
                balance = account_info.balance
            risk_amount = balance * self.risk_percentage
            sl_distance = abs(entry - sl) / self.point  # in points
            volume = self.calculate_volume(risk_amount, sl_distance)

            order_type = mt5.ORDER_TYPE_BUY if signal_type == "buy" else mt5.ORDER_TYPE_SELL
            h1_rsi, _ = self.get_rsi_signal("H1", h1_df)
            m15_rsi, _ = self.get_rsi_signal("M15", m15_df)
            m5_rsi, _ = self.get_rsi_signal("M5", m5_df)
            self.place_order(
                order_type=order_type,
                volume=volume,
                price=entry,
                sl=sl,
                tp=tp,
                strategy_name="MultiTimeframeRSI",
                rsi_values=(m5_rsi, m15_rsi, h1_rsi),
                macd_crossover=False,
                breakout_distance=strength,
            )
            self.logger.info(
                f"Placed {signal_type} order: Entry={entry:.5f}, SL={sl:.5f}, TP={tp:.5f}, Volume={volume}"
            )
            time.sleep(900)
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
        strategy = MultiTimeframeRSIStrategy(symbol="XAUUSD")
        m5_df = strategy.get_data("M5", 400)
        m15_df = strategy.get_data("M15", 200)
        h1_df = strategy.get_data("H1", 200)
        h4_df = strategy.get_data("H4", 100)
        d1_df = strategy.get_data("D1", 100)
        if all(df is not None for df in [m5_df, m15_df, h1_df, h4_df, d1_df]):
            signal, signal_type, strength = strategy.check_signals(m15_df, h4_df, d1_df, m5_df, h1_df)
            logging.info(f"Signal: {signal}, Type: {signal_type}, Strength: {strength}")
        else:
            logging.error("Failed to get data for testing")
    except Exception as e:
        logging.error(f"Error: {str(e)}")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()