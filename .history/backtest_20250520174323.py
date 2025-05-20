"""
Backtesting Module for Trading Strategies
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import os
import csv
import json

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from strategies.core.base_trading_strategy import BaseTradingStrategy
from strategies.core.utils import setup_logger
from strategies import MATradingStrategy, RSITradingStrategy, MultiTimeframeRSIStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("backtest/logs/backtest.log")],
)
logger = logging.getLogger(__name__)

BACKTEST_DAYS = 365
END_DATE = datetime(2025, 5, 18, 23, 59, 59)
MIN_BARS = 10
RISK_FREE_RATE = 0.02


class BacktestManager:
    """
    Manages and runs multiple trading strategies in parallel during backtesting.
    """
    
    def __init__(
        self,
        symbol: str,
        config_path: str,
        start_date: datetime,
        end_date: datetime,
        initial_balance: float = 100.0,
        leverage: int = 2000,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the backtest manager.
        
        Args:
            symbol: Trading symbol
            config_path: Path to configuration file
            start_date: Start date for backtesting
            end_date: End date for backtesting
            initial_balance: Initial balance for each strategy
            leverage: Trading leverage
            logger: Optional logger for logging events
        """
        self.symbol = symbol
        self.config_path = config_path
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.running = True  # Flag to control backtest execution
        
        # Initialize logger
        if logger is None:
            self.logger = self._setup_logger()
        else:
            self.logger = logger
            
        # Initialize MT5
        if not self.initialize_mt5():
            raise Exception("MT5 initialization failed")
            
        # Load historical data
        self.historical_data = self._load_historical_data()
        
        # Initialize strategies
        self.strategies = self._initialize_strategies()
        
        # Results storage
        self.results: Dict[str, Dict[str, Any]] = {}
        
        # Thread management
        self.threads: List[threading.Thread] = []
        
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the backtest manager."""
        logger = logging.getLogger("BacktestManager")
        logger.setLevel(logging.INFO)
        
        # Add console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger
        
    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection."""
        try:
            if not mt5.initialize():
                self.logger.error("MT5 initialization failed")
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
            
    def _load_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Load historical data for all timeframes."""
        data = {}
        timeframes = {
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
        # MT5 limits
        MAX_CANDLES = 67000  # Maximum candles that can be loaded at once
        
        # Calculate required days for each timeframe
        required_days = {
            'M5': 365 + 200,  # 1 year + 200 days for indicators
            'M15': 365 + 200,  # 1 year + 200 days for indicators
            'H1': 365 * 2,    # 2 years
            'H4': 365 * 2,    # 2 years
            'D1': 365 + 200   # 1 year + 200 days for indicators
        }
        
        # Calculate required bars for each timeframe
        required_bars = {
            'M5': required_days['M5'] * 24 * 12,  # 12 bars per hour * 24 hours * days
            'M15': required_days['M15'] * 24 * 4,  # 4 bars per hour * 24 hours * days
            'H1': required_days['H1'] * 24,       # 24 bars per day * days
            'H4': required_days['H4'] * 6,        # 6 bars per day * days
            'D1': required_days['D1']             # 1 bar per day * days
        }
        
        # Calculate maximum days we can load for each timeframe
        max_days = {
            'M5': MAX_CANDLES // (24 * 12),    # 12 bars per hour * 24 hours
            'M15': MAX_CANDLES // (24 * 4),    # 4 bars per hour * 24 hours
            'H1': MAX_CANDLES // 24,           # 24 bars per day
            'H4': MAX_CANDLES // 6,            # 6 bars per day
            'D1': MAX_CANDLES                  # 1 bar per day
        }
        
        self.logger.info(f"Required days for each timeframe:")
        for tf, days in required_days.items():
            self.logger.info(f"{tf}: {days} days")
        
        self.logger.info(f"Maximum days that can be loaded for each timeframe:")
        for tf, days in max_days.items():
            self.logger.info(f"{tf}: {days} days")
        
        # Create data directory structure
        data_dir = os.path.join('data', self.symbol)
        os.makedirs(data_dir, exist_ok=True)
        
        # Calculate start date to include additional days for indicators
        start_date = self.start_date - timedelta(days=200)
        self.logger.info(f"Loading data from {start_date} to {self.end_date} (including 200 days for indicators)")
        
        # First, try to load M5 and M15 data as they are critical
        critical_timeframes = ['M5', 'M15']
        for tf_name in critical_timeframes:
            try:
                tf = timeframes[tf_name]
                csv_file = os.path.join(data_dir, f"{tf_name}.csv")
                existing_data = None
                
                # Try to load existing data
                if os.path.exists(csv_file):
                    try:
                        self.logger.info(f"Loading {tf_name} data from {csv_file}")
                        existing_data = pd.read_csv(csv_file)
                        if 'time' not in existing_data.columns:
                            self.logger.error(f"Missing 'time' column in {tf_name} data")
                            existing_data = None
                        else:
                            existing_data['time'] = pd.to_datetime(existing_data['time'])
                            
                            # Check if we have enough data
                            if len(existing_data) >= required_bars[tf_name]:
                                self.logger.info(f"Using existing {tf_name} data with {len(existing_data)} bars")
                                data[tf_name] = existing_data
                                continue
                            else:
                                self.logger.info(f"Existing {tf_name} data has {len(existing_data)} bars")
                                # Calculate how many more days we need
                                days_needed = (required_bars[tf_name] - len(existing_data)) / (24 * (12 if tf_name == 'M5' else 4))
                                start_date = self.end_date - timedelta(days=min(days_needed, max_days[tf_name]))
                    except Exception as e:
                        self.logger.error(f"Error loading existing {tf_name} data: {str(e)}")
                        existing_data = None
                
                if existing_data is None:
                    self.logger.info(f"No existing {tf_name} data found or error loading, will download from {start_date}")
                    # Limit start date based on max days
                    start_date = self.end_date - timedelta(days=min(required_days[tf_name], max_days[tf_name]))
                
                # For M5 and M15, try loading in smaller chunks
                if tf_name in ['M5', 'M15']:
                    chunk_size = timedelta(days=30)  # Load 30 days at a time
                    current_start = start_date
                    all_rates = []
                    
                    while current_start < self.end_date:
                        current_end = min(current_start + chunk_size, self.end_date)
                        self.logger.info(f"Fetching {tf_name} data chunk from {current_start} to {current_end}")
                        
                        try:
                            rates = mt5.copy_rates_range(
                                self.symbol,
                                tf,
                                current_start,
                                current_end
                            )
                            
                            if rates is not None and len(rates) > 0:
                                all_rates.extend(rates)
                                self.logger.info(f"Loaded {len(rates)} {tf_name} bars for chunk")
                            else:
                                self.logger.warning(f"No data received for {tf_name} chunk from {current_start} to {current_end}")
                        except Exception as e:
                            self.logger.error(f"Error fetching {tf_name} chunk: {str(e)}")
                        
                        current_start = current_end
                    
                    if all_rates:
                        try:
                            new_data = pd.DataFrame(all_rates)
                            if 'time' not in new_data.columns:
                                raise ValueError(f"Missing 'time' column in {tf_name} data from MT5")
                            new_data['time'] = pd.to_datetime(new_data['time'], unit='s')
                        except Exception as e:
                            self.logger.error(f"Error processing {tf_name} data: {str(e)}")
                            if existing_data is not None:
                                data[tf_name] = existing_data
                            continue
                    else:
                        self.logger.error(f"No {tf_name} data received from MT5")
                        if existing_data is not None:
                            data[tf_name] = existing_data
                        continue
                else:
                    # For other timeframes, load normally
                    self.logger.info(f"Fetching {tf_name} data from MT5 from {start_date} to {self.end_date}")
                    try:
                        rates = mt5.copy_rates_range(
                            self.symbol,
                            tf,
                            start_date,
                            self.end_date
                        )
                        
                        if rates is None or len(rates) == 0:
                            self.logger.error(f"No data received from MT5 for {self.symbol} {tf_name}")
                            if existing_data is not None:
                                data[tf_name] = existing_data
                            continue
                        
                        new_data = pd.DataFrame(rates)
                        if 'time' not in new_data.columns:
                            raise ValueError(f"Missing 'time' column in {tf_name} data from MT5")
                        new_data['time'] = pd.to_datetime(new_data['time'], unit='s')
                    except Exception as e:
                        self.logger.error(f"Error fetching {tf_name} data: {str(e)}")
                        if existing_data is not None:
                            data[tf_name] = existing_data
                        continue
                
                # Combine with existing data if available
                if existing_data is not None:
                    try:
                        combined_data = pd.concat([existing_data, new_data])
                        combined_data = combined_data.drop_duplicates(subset=['time'])
                        combined_data = combined_data.sort_values('time')
                        data[tf_name] = combined_data
                    except Exception as e:
                        self.logger.error(f"Error combining {tf_name} data: {str(e)}")
                        data[tf_name] = new_data
                else:
                    data[tf_name] = new_data
                
                # Log the actual number of bars loaded
                self.logger.info(f"Loaded {len(data[tf_name])} {tf_name} bars")
                
                # Save to CSV
                try:
                    data[tf_name].to_csv(csv_file, index=False)
                    self.logger.info(f"Saved {len(data[tf_name])} {tf_name} bars to {csv_file}")
                except Exception as e:
                    self.logger.error(f"Error saving {tf_name} data to CSV: {str(e)}")
                
            except Exception as e:
                self.logger.error(f"Error loading {tf_name} data: {str(e)}")
                if existing_data is not None:
                    data[tf_name] = existing_data
                continue
        
        # Verify critical timeframes are loaded
        if not all(tf in data for tf in critical_timeframes):
            missing = [tf for tf in critical_timeframes if tf not in data]
            raise Exception(f"Failed to load critical timeframe data: {', '.join(missing)}")
        
        # Load other timeframes
        for tf_name, tf in timeframes.items():
            if tf_name in critical_timeframes:
                continue
            
            try:
                csv_file = os.path.join(data_dir, f"{tf_name}.csv")
                existing_data = None
                
                # Try to load existing data
                if os.path.exists(csv_file):
                    try:
                        self.logger.info(f"Loading {tf_name} data from {csv_file}")
                        existing_data = pd.read_csv(csv_file)
                        if 'time' not in existing_data.columns:
                            self.logger.error(f"Missing 'time' column in {tf_name} data")
                            existing_data = None
                        else:
                            existing_data['time'] = pd.to_datetime(existing_data['time'])
                            
                            # Check if we have enough data
                            if len(existing_data) >= required_bars[tf_name]:
                                self.logger.info(f"Using existing {tf_name} data with {len(existing_data)} bars")
                                data[tf_name] = existing_data
                                continue
                            else:
                                self.logger.info(f"Existing {tf_name} data has {len(existing_data)} bars")
                                # Calculate how many more days we need
                                days_needed = (required_bars[tf_name] - len(existing_data)) / (24 * (6 if tf_name == 'H4' else 1))
                                start_date = self.end_date - timedelta(days=min(days_needed, max_days[tf_name]))
                    except Exception as e:
                        self.logger.error(f"Error loading existing {tf_name} data: {str(e)}")
                        existing_data = None
                
                if existing_data is None:
                    self.logger.info(f"No existing {tf_name} data found or error loading, will download from {start_date}")
                    # Limit start date based on max days
                    start_date = self.end_date - timedelta(days=min(required_days[tf_name], max_days[tf_name]))
                
                # Fetch new data from MT5
                self.logger.info(f"Fetching {tf_name} data from MT5 from {start_date} to {self.end_date}")
                try:
                    rates = mt5.copy_rates_range(
                        self.symbol,
                        tf,
                        start_date,
                        self.end_date
                    )
                    
                    if rates is None or len(rates) == 0:
                        self.logger.error(f"No data received from MT5 for {self.symbol} {tf_name}")
                        if existing_data is not None:
                            data[tf_name] = existing_data
                        continue
                    
                    new_data = pd.DataFrame(rates)
                    if 'time' not in new_data.columns:
                        raise ValueError(f"Missing 'time' column in {tf_name} data from MT5")
                    new_data['time'] = pd.to_datetime(new_data['time'], unit='s')
                except Exception as e:
                    self.logger.error(f"Error fetching {tf_name} data: {str(e)}")
                    if existing_data is not None:
                        data[tf_name] = existing_data
                    continue
                
                # Combine with existing data if available
                if existing_data is not None:
                    try:
                        combined_data = pd.concat([existing_data, new_data])
                        combined_data = combined_data.drop_duplicates(subset=['time'])
                        combined_data = combined_data.sort_values('time')
                        data[tf_name] = combined_data
                    except Exception as e:
                        self.logger.error(f"Error combining {tf_name} data: {str(e)}")
                        data[tf_name] = new_data
                else:
                    data[tf_name] = new_data
                
                # Log the actual number of bars loaded
                self.logger.info(f"Loaded {len(data[tf_name])} {tf_name} bars")
                
                # Save to CSV
                try:
                    data[tf_name].to_csv(csv_file, index=False)
                    self.logger.info(f"Saved {len(data[tf_name])} {tf_name} bars to {csv_file}")
                except Exception as e:
                    self.logger.error(f"Error saving {tf_name} data to CSV: {str(e)}")
                
            except Exception as e:
                self.logger.error(f"Error loading {tf_name} data: {str(e)}")
                if existing_data is not None:
                    data[tf_name] = existing_data
                continue
        
        if not data:
            raise Exception("No data loaded for any timeframe")
        
        return data
        
    def _initialize_strategies(self) -> Dict[str, Any]:
        """Initialize all trading strategies."""
        strategies = {}
        
        # Initialize RSI Strategy
        # strategies['RSI'] = RSITradingStrategy(
        #     symbol=self.symbol,
        #     config_path=self.config_path,
        #     mt5_initialized=True,
        #     logger=self.logger,
        #     initial_balance=self.initial_balance,
        #     leverage=self.leverage
        # )
        
        # # Initialize MA Strategy
        # strategies['MA'] = MATradingStrategy(
        #     symbol=self.symbol,
        #     config_path=self.config_path,
        #     mt5_initialized=True,
        #     logger=self.logger,
        #     initial_balance=self.initial_balance,
        #     leverage=self.leverage
        # )
        
        # Initialize Multi-Timeframe RSI Strategy
        strategies['MultiTimeframeRSI'] = MultiTimeframeRSIStrategy(
            symbol=self.symbol,
            config_path=self.config_path,
            mt5_initialized=True,
            logger=self.logger,
            initial_balance=self.initial_balance,
            leverage=self.leverage
        )
        
        return strategies
        
    def _run_backtest(self, strategy_name: str, strategy: Any):
        """Run backtest for a single strategy."""
        try:
            self.logger.info(f"Starting backtest for {strategy_name}")
            
            # Get required timeframes for this strategy
            m15_df = self.historical_data['M15']
            h4_df = self.historical_data['H4']
            d1_df = self.historical_data['D1']
            m5_df = self.historical_data.get('M5')
            h1_df = self.historical_data.get('H1')
            
            # Validate data
            for df_name, df in [('M15', m15_df), ('H4', h4_df), ('D1', d1_df), ('M5', m5_df), ('H1', h1_df)]:
                if df is not None:
                    self.logger.info(f"{df_name} data shape: {df.shape}")
                    self.logger.info(f"{df_name} columns: {df.columns.tolist()}")
                    if 'close' not in df.columns:
                        self.logger.error(f"Missing 'close' column in {df_name} data")
                        return
                    if len(df) == 0:
                        self.logger.error(f"Empty DataFrame for {df_name}")
                        return
                elif df_name in ['M5', 'M15', 'H1'] and strategy_name == 'MultiTimeframeRSI':
                    self.logger.error(f"Missing required {df_name} data for MultiTimeframeRSI strategy")
                    return
            
            # Initialize results storage
            self.results[strategy_name] = {
                'trades': [],
                'balance': self.initial_balance,
                'equity': self.initial_balance,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
            
            # Track current day for logging
            current_day = None
            
            # Log data ranges
            self.logger.info(f"Data range for {strategy_name}:")
            self.logger.info(f"M5: {m5_df['time'].min()} to {m5_df['time'].max()}" if m5_df is not None else "M5: No data")
            self.logger.info(f"M15: {m15_df['time'].min()} to {m15_df['time'].max()}")
            self.logger.info(f"H1: {h1_df['time'].min()} to {h1_df['time'].max()}" if h1_df is not None else "H1: No data")
            self.logger.info(f"H4: {h4_df['time'].min()} to {h4_df['time'].max()}")
            self.logger.info(f"D1: {d1_df['time'].min()} to {d1_df['time'].max()}")
            
            # Run backtest
            for i in range(len(m15_df)):
                if not self.running:  # Check if we should stop
                    self.logger.info(f"Stopping backtest for {strategy_name}")
                    break
                    
                # Get current data
                current_m15 = m15_df.iloc[i]
                
                # Log new day
                current_date = current_m15['time'].date()
                if current_day != current_date:
                    current_day = current_date
                    self.logger.info(f"Processing data for {strategy_name} - Date: {current_date}")
                
                # Get historical data up to current time
                current_time = current_m15['time']
                
                # Filter data for each timeframe up to current time
                m15_data = m15_df[m15_df['time'] <= current_time]
                h4_data = h4_df[h4_df['time'] <= current_time]
                d1_data = d1_df[d1_df['time'] <= current_time]
                m5_data = m5_df[m5_df['time'] <= current_time] if m5_df is not None else None
                h1_data = h1_df[h1_df['time'] <= current_time] if h1_df is not None else None
                
                # Validate filtered data
                if len(m15_data) == 0 or len(h4_data) == 0 or len(d1_data) == 0:
                    self.logger.warning(f"Empty data after filtering at {current_time}")
                    continue
                
                if strategy_name == 'MultiTimeframeRSI' and (m5_data is None or len(m5_data) == 0 or h1_data is None or len(h1_data) == 0):
                    self.logger.warning(f"Missing required timeframe data at {current_time}")
                    continue
                
                # Get the latest data point for each timeframe
                current_h4 = h4_data.iloc[-1]
                current_d1 = d1_data.iloc[-1]
                current_m5 = m5_data.iloc[-1] if m5_data is not None and len(m5_data) > 0 else None
                current_h1 = h1_data.iloc[-1] if h1_data is not None and len(h1_data) > 0 else None
                
                try:
                    # Check for signals
                    signal_found, signal_type, strength = strategy.check_signals(
                        m15_data,
                        h4_data,
                        d1_data,
                        m5_data,
                        h1_data
                    )
                    
                    if signal_found:
                        # Calculate position size
                        volume = strategy.calculate_volume(
                            current_m15['close'],
                            current_m15['close'] - strategy.stop_loss_points * mt5.symbol_info(self.symbol).point
                        )
                        
                        # Simulate trade
                        trade_result = self._simulate_trade(
                            strategy_name,
                            signal_type,
                            volume,
                            current_m15['close'],
                            current_m15['time']
                        )
                        
                        # Update results
                        self.results[strategy_name]['trades'].append(trade_result)
                        self._update_statistics(strategy_name)
                        
                except Exception as e:
                    self.logger.error(f"Error in check_signals: {str(e)}")
                    self.logger.error(f"Current time: {current_time}")
                    self.logger.error(f"M5 data shape: {m5_data.shape if m5_data is not None else 'None'}")
                    self.logger.error(f"M15 data shape: {m15_data.shape}")
                    self.logger.error(f"H1 data shape: {h1_data.shape if h1_data is not None else 'None'}")
                    self.logger.error(f"H4 data shape: {h4_data.shape}")
                    self.logger.error(f"D1 data shape: {d1_data.shape}")
                    continue
                
            self.logger.info(f"Completed backtest for {strategy_name}")
            
        except Exception as e:
            self.logger.error(f"Error in {strategy_name} backtest: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
    def _simulate_trade(
        self,
        strategy_name: str,
        signal_type: str,
        volume: float,
        price: float,
        time: datetime
    ) -> Dict[str, Any]:
        """Simulate a trade and return its result."""
        # Calculate stop loss and take profit
        point = mt5.symbol_info(self.symbol).point
        strategy = self.strategies[strategy_name]
        
        if signal_type == 'BUY':
            sl = price - strategy.stop_loss_points * point
            tp = price + strategy.take_profit_points * point
        else:  # SELL
            sl = price + strategy.stop_loss_points * point
            tp = price - strategy.take_profit_points * point
            
        # Simulate trade outcome
        # For simplicity, we'll use a random outcome based on win rate
        import random
        win_rate = 0.6  # 60% win rate
        is_win = random.random() < win_rate
        
        if is_win:
            profit = abs(tp - price) * volume
        else:
            profit = -abs(sl - price) * volume
            
        return {
            'time': time,
            'type': signal_type,
            'volume': volume,
            'price': price,
            'sl': sl,
            'tp': tp,
            'profit': profit,
            'is_win': is_win
        }
        
    def _update_statistics(self, strategy_name: str):
        """Update backtest statistics for a strategy."""
        trades = self.results[strategy_name]['trades']
        if not trades:
            return
            
        # Calculate statistics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['is_win']])
        total_profit = sum(t['profit'] for t in trades)
        total_loss = abs(sum(t['profit'] for t in trades if not t['is_win']))
        
        # Update results
        self.results[strategy_name].update({
            'balance': self.initial_balance + total_profit,
            'equity': self.initial_balance + total_profit,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'profit_factor': total_profit / total_loss if total_loss > 0 else float('inf')
        })
        
    def _save_results_to_csv(self, strategy_name: str):
        """Save backtest results to CSV files."""
        try:
            # Create results directory if it doesn't exist
            results_dir = os.path.join('backtest', 'results')
            os.makedirs(results_dir, exist_ok=True)
            
            # Save detailed trades to CSV
            trades_file = os.path.join(results_dir, f'backtest_results_{strategy_name}.csv')
            trades = self.results[strategy_name]['trades']
            
            if trades:
                # Convert trades to detailed format
                detailed_trades = []
                for trade in trades:
                    detailed_trade = {
                        'entry_time': trade['time'],
                        'entry_price': trade['price'],
                        'volume': trade['volume'],
                        'sl': trade['sl'],
                        'tp': trade['tp'],
                        'order_type': trade['type'],
                        'exit_time': trade['time'] + timedelta(minutes=15),  # Simulated exit time
                        'exit_price': trade['tp'] if trade['is_win'] else trade['sl'],  # Simulated exit price
                        'profit': trade['profit'],
                        'status': 'closed',
                        'strategy': strategy_name,
                        'priority': 'medium',  # Default priority
                        'leverage': self.leverage
                    }
                    detailed_trades.append(detailed_trade)
                
                trades_df = pd.DataFrame(detailed_trades)
                trades_df.to_csv(trades_file, index=False)
                self.logger.info(f"Saved {len(trades)} trades to {trades_file}")
            
            # Save summary to CSV
            summary_file = os.path.join(results_dir, f'{strategy_name}_summary.csv')
            summary = {
                'strategy': strategy_name,
                'initial_balance': self.initial_balance,
                'final_balance': self.results[strategy_name]['balance'],
                'total_profit': self.results[strategy_name]['balance'] - self.initial_balance,
                'win_rate': self.results[strategy_name]['win_rate'],
                'profit_factor': self.results[strategy_name]['profit_factor'],
                'total_trades': len(trades),
                'max_drawdown': self.results[strategy_name]['max_drawdown'],
                'leverage': self.leverage
            }
            
            summary_df = pd.DataFrame([summary])
            summary_df.to_csv(summary_file, index=False)
            self.logger.info(f"Saved summary to {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving results for {strategy_name}: {str(e)}")
            
    def _print_results(self):
        """Print backtest results for all strategies and save to CSV."""
        print("\nBacktest Results:")
        print("=" * 80)
        
        for name, result in self.results.items():
            print(f"\nStrategy: {name}")
            print("-" * 40)
            print(f"Initial Balance: ${self.initial_balance:.2f}")
            print(f"Final Balance: ${result['balance']:.2f}")
            print(f"Total Profit: ${result['balance'] - self.initial_balance:.2f}")
            print(f"Win Rate: {result['win_rate']*100:.1f}%")
            print(f"Profit Factor: {result['profit_factor']:.2f}")
            print(f"Total Trades: {len(result['trades'])}")
            
            # Save results to CSV
            self._save_results_to_csv(name)
            
    def run_backtest(self):
        """Run backtest for all strategies in parallel."""
        try:
            # Start each strategy in a separate thread
            for name, strategy in self.strategies.items():
                thread = threading.Thread(
                    target=self._run_backtest,
                    args=(name, strategy),
                    name=f"Thread-{name}"
                )
                self.threads.append(thread)
                thread.start()
                self.logger.info(f"Started backtest for {name}")
                
            # Wait for all threads to complete or handle Ctrl+C
            while any(thread.is_alive() for thread in self.threads):
                time.sleep(0.1)  # Small sleep to prevent CPU overuse
                
        except KeyboardInterrupt:
            self.logger.info("Received stop signal, stopping backtest...")
            self.running = False  # Signal all threads to stop
            
            # Wait for threads to finish
            for thread in self.threads:
                thread.join(timeout=5.0)  # Wait up to 5 seconds for each thread
                
        finally:
            # Print and save results even if interrupted
            self._print_results()
            self.logger.info("Backtest completed and results saved")
            
def main():
    """Run backtest for all strategies."""
    try:
        # Calculate start date based on END_DATE and BACKTEST_DAYS
        start_date = END_DATE - timedelta(days=BACKTEST_DAYS)
        
        # Initialize backtest manager
        manager = BacktestManager(
            symbol="XAUUSDm",
            config_path="backtest/config/backtest_config.json",
            start_date=start_date,
            end_date=END_DATE,
            initial_balance=100.0,
            leverage=2000
        )
        
        # Run backtest
        manager.run_backtest()
        
    except KeyboardInterrupt:
        print("\nBacktest interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Shutdown MT5
        mt5.shutdown()
        print("\nBacktest process completed")
        
if __name__ == "__main__":
    main()