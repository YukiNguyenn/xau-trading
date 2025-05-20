"""
Base trading strategy module.

This module provides the base class for all trading strategies.
"""

import logging
import json
import MetaTrader5 as mt5
import pandas as pd
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import os
import time

from .utils import setup_logger
from .indicators import calculate_rsi, calculate_macd, calculate_ma
from .trade_manager import TradeManager
from .risk_manager import RiskManager
from .data_manager import DataManager


class BaseTradingStrategy:
    """
    Base class for all trading strategies.
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
        Initialize the base trading strategy.
        
        Args:
            symbol: Trading symbol
            config_path: Path to configuration file
            mt5_initialized: Whether MT5 is already initialized
            logger: Optional logger for logging strategy events
            use_mock_data: Whether to use mock data (for backtesting)
            initial_balance: Initial balance to use (for backtesting)
            leverage: Trading leverage (default: 2000)
        """
        self.symbol = symbol
        self.config_path = config_path
        self.mt5_initialized = mt5_initialized
        self.use_mock_data = use_mock_data
        self.initial_balance = initial_balance
        self.leverage = leverage
        
        # Initialize logger
        if logger is None:
            self.logger = setup_logger(self.__class__.__name__)
        else:
            self.logger = logger
            
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Initialize MT5 if not already initialized
        if not mt5_initialized:
            if not mt5.initialize():
                raise Exception("Failed to initialize MT5")
                
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise Exception(f"Failed to get symbol info for {symbol}")
            
        self.point = symbol_info.point
        self.min_volume = symbol_info.volume_min
        self.max_volume = symbol_info.volume_max
        
        # Initialize trade log file
        self.trade_log_file = f"trades/{symbol}_trades.csv"
        self.initialize_trade_log()
        
        # Initialize trade manager
        self.trade_manager = TradeManager(symbol, logger=self.logger)
        
        self.logger.info(f"Initialized {self.__class__.__name__} with {leverage}x leverage")
        
        # Trading parameters from config
        trading_config = self.config.get("trading", {})
        self.stop_loss_points = trading_config.get("stop_loss_points", 100)
        self.take_profit_points = trading_config.get("take_profit_points", 200)
        self.max_open_positions = trading_config.get("max_open_positions", 3)
        self.commission = trading_config.get("commission", 0.0001)
        
        # Timeframes mapping
        self.timeframes = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1,
        }
        
        # Initialize managers
        self.risk_manager = RiskManager(
            account_balance=self.initial_balance,
            logger=self.logger
        )
        
        # Load trading parameters from config
        self.spread_points = self.config['trading']['spread_points']
        self.trailing_stop_points = self.config['risk']['trailing_stop_points']
        self.priority_levels = self.config['position']['priority_levels']
        
    def get_data(self, timeframe: str, bars: int, retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Fetch historical data from MT5 with retries.
        
        Args:
            timeframe: Timeframe to fetch data for (e.g., 'M15', 'H4', 'D1')
            bars: Number of bars to fetch
            retries: Number of retry attempts
            
        Returns:
            Optional[pd.DataFrame]: DataFrame containing price data or None if failed
        """
        if timeframe not in self.timeframes:
            self.logger.error(f"Invalid timeframe: {timeframe}")
            return None
            
        for attempt in range(retries):
            try:
                # Calculate number of bars to request based on timeframe
                requested_bars = bars
                if timeframe == 'M15':
                    requested_bars = max(bars * 4, 2000)  # Request at least 2000 bars for M15
                elif timeframe == 'H1':
                    requested_bars = max(bars * 3, 500)  # Request at least 500 bars for H1
                elif timeframe == 'H4':
                    requested_bars = max(bars * 2, 300)  # Request at least 300 bars for H4
                elif timeframe == 'D1':
                    requested_bars = max(bars * 2, 200)  # Request at least 200 bars for D1

                self.logger.info(f"Requesting {requested_bars} bars for {timeframe} (minimum required: {bars})")
                rates = mt5.copy_rates_from_pos(self.symbol, self.timeframes[timeframe], 0, requested_bars)
                
                if rates is None or len(rates) == 0:
                    self.logger.warning(f"No data received for {self.symbol} {timeframe} on attempt {attempt + 1}")
                    if attempt == retries - 1:
                        return None
                    time.sleep(1)
                    continue
                    
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                
                # Log the actual number of bars received
                self.logger.info(f"Received {len(df)} bars for {timeframe} (requested: {requested_bars}, required: {bars})")
                
                # Ensure we have enough bars
                if len(df) < bars:
                    self.logger.warning(f"Not enough data for {timeframe}: got {len(df)} bars, need {bars}")
                    if attempt == retries - 1:
                        return None
                    time.sleep(1)
                    continue
                    
                return df
                
            except Exception as e:
                self.logger.error(f"Error retrieving {timeframe} data on attempt {attempt + 1}: {str(e)}")
                if attempt == retries - 1:
                    return None
                time.sleep(1)
                
        return None
        
    def _generate_mock_data(self, timeframe: str, bars: int) -> pd.DataFrame:
        """
        Generate mock price data for testing when MT5 data is unavailable.
        
        Args:
            timeframe: Timeframe to generate data for
            bars: Number of bars to generate
            
        Returns:
            pd.DataFrame: DataFrame containing mock price data
        """
        dates = pd.date_range(start='2024-01-01', periods=bars, freq='15min')
        data = pd.DataFrame({
            'time': dates,
            'open': np.random.normal(2000, 10, bars),
            'high': np.random.normal(2005, 10, bars),
            'low': np.random.normal(1995, 10, bars),
            'close': np.random.normal(2000, 10, bars),
            'tick_volume': np.random.randint(100, 1000, bars)
        })
        return data
        
    def _setup_logger(self, name: str) -> logging.Logger:
        """
        Set up logger for the strategy.
        
        Args:
            name: Logger name
            
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def initialize_trade_log(self) -> None:
        """
        Initialize the trade log file with headers if it doesn't exist.
        """
        try:
            if not os.path.exists(self.trade_log_file):
                with open(self.trade_log_file, 'w') as f:
                    f.write("timestamp,symbol,order_type,volume,entry_price,sl,tp,strategy_name,rsi_short,rsi_medium,rsi_long,macd_crossover,breakout_distance\n")
                self.logger.info(f"Created trade log file: {self.trade_log_file}")
        except Exception as e:
            self.logger.error(f"Error initializing trade log: {str(e)}")
            
    def log_trade(
        self,
        order_type: str,
        volume: float,
        entry_price: float,
        sl: float,
        tp: float,
        rsi_values: Tuple[Optional[float], Optional[float], Optional[float]] = (None, None, None),
        macd_crossover: bool = False,
        breakout_distance: float = 0.0
    ) -> None:
        """
        Log a trade to the trade log file.
        
        Args:
            order_type: Type of order (BUY/SELL)
            volume: Trade volume
            entry_price: Entry price
            sl: Stop loss
            tp: Take profit
            rsi_values: Tuple of RSI values (short, medium, long)
            macd_crossover: Whether MACD crossover occurred
            breakout_distance: Distance from breakout level
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rsi_short, rsi_medium, rsi_long = rsi_values
            
            with open(self.trade_log_file, 'a') as f:
                f.write(
                    f"{timestamp},{self.symbol},{order_type},{volume},"
                    f"{entry_price},{sl},{tp},{self.__class__.__name__},"
                    f"{rsi_short},{rsi_medium},{rsi_long},"
                    f"{macd_crossover},{breakout_distance}\n"
                )
            self.logger.info(f"Trade logged to {self.trade_log_file}")
        except Exception as e:
            self.logger.error(f"Error logging trade: {str(e)}")
            
    def initialize_mt5(self) -> bool:
        """
        Initialize MT5 connection.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            if not mt5.initialize():
                self.logger.error("MT5 initialization failed")
                return False
                
            # Only show account info when not in backtest mode and MT5 is not already initialized
            if not hasattr(self, 'use_mock_data') or not self.use_mock_data:
                account_info = mt5.account_info()
                if account_info:
                    self.logger.info(
                        f"MT5 initialized successfully\n"
                        f"Login: {account_info.login}\n"
                        f"Server: {account_info.server}\n"
                        f"Balance: {account_info.balance}\n"
                        f"Equity: {account_info.equity}\n"
                        f"Margin: {account_info.margin}\n"
                        f"Free Margin: {account_info.margin_free}\n"
                        f"Margin Level: {account_info.margin_level}%"
                    )
                else:
                    self.logger.error("Failed to get account info")
            else:
                self.logger.info("MT5 initialized successfully")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing MT5: {str(e)}")
            return False
            
    def check_signals(
        self,
        m15_df: pd.DataFrame,
        h4_df: pd.DataFrame,
        d1_df: pd.DataFrame,
        m5_df: Optional[pd.DataFrame] = None,
        h1_df: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, Optional[str], float]:
        """
        Check for trading signals.
        
        Args:
            m15_df: M15 timeframe data
            h4_df: H4 timeframe data
            d1_df: D1 timeframe data
            m5_df: Optional M5 timeframe data
            h1_df: Optional H1 timeframe data
            
        Returns:
            Tuple[bool, Optional[str], float]: (signal_found, signal_type, strength)
        """
        raise NotImplementedError("Subclasses must implement check_signals()")
        
    def run_strategy(self) -> None:
        """
        Run the trading strategy.
        """
        try:
            # Fetch data for all timeframes
            m15_df = self.data_manager.fetch_data('M15')
            h4_df = self.data_manager.fetch_data('H4')
            d1_df = self.data_manager.fetch_data('D1')
            
            if m15_df is None or h4_df is None or d1_df is None:
                self.logger.error("Failed to fetch data")
                return
                
            # Check for signals
            signal_found, signal_type, strength = self.check_signals(
                m15_df, h4_df, d1_df
            )
            
            if signal_found and signal_type:
                self.logger.info(
                    f"Signal found: {signal_type} (strength: {strength:.2f})"
                )
                
                # Get current price
                tick = mt5.symbol_info_tick(self.symbol)
                if tick is None:
                    self.logger.error("Failed to get current price")
                    return
                    
                # Calculate entry price
                entry_price = tick.ask if signal_type == 'buy' else tick.bid
                
                # Calculate stop loss and take profit
                atr = self.data_manager.calculate_atr(m15_df).iloc[-1]
                sl = self.risk_manager.calculate_stop_loss(
                    entry_price,
                    signal_type,
                    atr
                )
                tp = self.risk_manager.calculate_take_profit(entry_price, sl)
                
                # Calculate position size
                volume = self.risk_manager.calculate_position_size(
                    entry_price,
                    sl
                )
                
                # Place order
                order_type = (
                    mt5.ORDER_TYPE_BUY if signal_type == 'buy'
                    else mt5.ORDER_TYPE_SELL
                )
                
                self.trade_manager.place_order(
                    order_type=order_type,
                    volume=volume,
                    price=entry_price,
                    sl=sl,
                    tp=tp,
                    strategy_name=self.__class__.__name__
                )
                
        except Exception as e:
            self.logger.error(f"Error running strategy: {str(e)}")
            
    def cleanup(self) -> None:
        """
        Clean up resources.
        """
        try:
            mt5.shutdown()
            self.logger.info("MT5 connection closed")
        except Exception as e:
            self.logger.error(f"Error cleaning up: {str(e)}")

    def _get_priority(
        self,
        rsi_short: Optional[float],
        rsi_medium: Optional[float],
        rsi_long: Optional[float],
        macd_crossover: bool,
        breakout_distance: float
    ) -> str:
        """
        Determine trade priority based on indicators.
        
        Args:
            rsi_short: Short-term RSI value
            rsi_medium: Medium-term RSI value
            rsi_long: Long-term RSI value
            macd_crossover: Whether MACD crossover occurred
            breakout_distance: Distance from breakout level
            
        Returns:
            str: Priority level ('high', 'medium', or 'low')
        """
        if not all(x is not None for x in [rsi_short, rsi_medium, rsi_long]):
            return 'low'
            
        # Check high priority conditions
        high_conditions = self.priority_levels['high']
        if (rsi_short >= high_conditions['rsi_short'] and
            rsi_medium >= high_conditions['rsi_medium'] and
            rsi_long >= high_conditions['rsi_long'] and
            macd_crossover == high_conditions['macd_crossover']):
            return 'high'
            
        # Check medium priority conditions
        medium_conditions = self.priority_levels['medium']
        if (rsi_short >= medium_conditions['rsi_short'] and
            rsi_medium >= medium_conditions['rsi_medium'] and
            rsi_long >= medium_conditions['rsi_long'] and
            macd_crossover == medium_conditions['macd_crossover']):
            return 'medium'
            
        # Default to low priority
        return 'low'

    def _load_config(self) -> Dict:
        """
        Load configuration from JSON file.
        
        Returns:
            Dict: Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config from {self.config_path}: {str(e)}")
            raise