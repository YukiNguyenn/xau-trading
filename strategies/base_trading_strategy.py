"""
Base Trading Strategy

This module provides the base class for all trading strategies.
"""

import logging
import os
import json
import time
from typing import Optional, Tuple, Dict, Any, List
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

class BaseTradingStrategy:
    """
    Base class for all trading strategies.
    Provides common functionality for MT5 connection, data fetching, and trade execution.
    """
    
    def __init__(
        self,
        symbol: str,
        config_path: str,
        mt5_initialized: bool = False,
        logger: Optional[logging.Logger] = None,
        use_mock_data: bool = False,
        initial_balance: Optional[float] = None,
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
        """
        self.symbol = symbol
        self.config_path = config_path
        self.use_mock_data = use_mock_data
        self.initial_balance = initial_balance
        
        # Initialize logger
        if logger is None:
            self.logger = self._setup_logger(self.__class__.__name__)
        else:
            self.logger = logger
            
        # Load configuration
        self.config = self._load_config()
        
        # Initialize MT5 if not already initialized
        if not mt5_initialized and not self.initialize_mt5():
            raise Exception("MT5 initialization failed")
            
        # Get account info
        account_info = mt5.account_info()
        if account_info is not None:
            self.balance = account_info.balance
            self.logger.info(f"Account Balance: {self.balance}")
        else:
            self.balance = initial_balance if initial_balance is not None else 10000
            self.logger.warning(f"Could not get account info, using default balance: {self.balance}")
            
        # Trading parameters from config
        self.min_volume = self.config['trading']['min_volume']
        self.max_volume = self.config['trading']['max_volume']
        self.stop_loss_points = self.config['trading']['stop_loss_points']
        self.take_profit_points = self.config['trading']['take_profit_points']
        
        # Timeframes mapping
        self.timeframes = {
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
        
        # Track open trades
        self.open_trades = []

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

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            raise

    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection."""
        try:
            if not mt5.initialize():
                self.logger.error("MT5 initialization failed")
                return False
                
            # Login to MT5
            if not mt5.login(
                login=self.config['mt5']['login'],
                password=self.config['mt5']['password'],
                server=self.config['mt5']['server']
            ):
                self.logger.error("MT5 login failed")
                return False
                
            # Get account info
            account_info = mt5.account_info()
            if account_info is not None:
                self.logger.info(f"MT5 initialized successfully")
                self.logger.info(f"Login: {account_info.login}")
                self.logger.info(f"Server: {account_info.server}")
                self.logger.info(f"Balance: {account_info.balance}")
                self.logger.info(f"Equity: {account_info.equity}")
                self.logger.info(f"Margin: {account_info.margin}")
                self.logger.info(f"Free Margin: {account_info.margin_free}")
                self.logger.info(f"Margin Level: {account_info.margin_level}%")
            else:
                self.logger.error("Failed to get account info")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing MT5: {str(e)}")
            return False

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

    def get_data(self, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
        """Fetch historical data from MT5."""
        try:
            if timeframe not in self.timeframes:
                self.logger.error(f"Invalid timeframe: {timeframe}")
                return None
                
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframes[timeframe], 0, bars)
            if rates is None or len(rates) == 0:
                self.logger.warning(f"No data for {self.symbol} {timeframe}")
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting data: {str(e)}")
            return None

    def place_order(
        self,
        order_type: int,
        volume: float,
        price: float,
        sl: float,
        tp: float,
        strategy_name: str,
        rsi_values: Optional[Tuple[float, float, float]] = None,
        macd_crossover: bool = False,
        breakout_distance: float = 0.0
    ) -> bool:
        """Place a trade order."""
        try:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 234000,
                "comment": strategy_name,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order failed: {result.comment}")
                return False
                
            self.logger.info(
                f"Order placed: {order_type}, Volume: {volume}, "
                f"Price: {price}, SL: {sl}, TP: {tp}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
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
        To be implemented by child classes.
        
        Returns:
            Tuple of (signal_found, signal_type, strength)
        """
        raise NotImplementedError("Child classes must implement check_signals")

    def run_strategy(self) -> None:
        """
        Main strategy execution loop.
        To be implemented by child classes.
        """
        raise NotImplementedError("Child classes must implement run_strategy")