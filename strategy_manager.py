"""
Strategy Manager

This module provides a centralized manager for running multiple trending strategies
concurrently in separate threads. It handles strategy initialization, data fetching,
execution, and proper cleanup of resources.
"""

import threading
import time
import logging
from typing import List, Type, Optional, Dict, Any
import MetaTrader5 as mt5
import pandas as pd
from strategies.core import setup_logger, BaseTradingStrategy
from strategies.core.trade_manager import TradeManager
from strategies.core.utils import setup_logger
import json
from strategies.rsi_trading_strategy import RSITradingStrategy
from strategies.ma_trading_strategy import MATradingStrategy
from strategies.multi_timeframe_rsi_strategy import MultiTimeframeRSIStrategy
import os
import csv
from datetime import datetime, timedelta


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("strategy_manager.log")
    ]
)
logger = logging.getLogger(__name__)


class StrategyManager:
    """
    Manages multiple trending strategies running in parallel.
    
    This class handles the lifecycle of multiple trending strategies, including
    initialization, data fetching, execution in separate threads, and proper shutdown.
    """
    
    def __init__(self, symbol: str = "XAUUSD", is_backtest: bool = False, leverage: int = 2000):
        self.symbol = symbol
        self.strategies = []
        self.threads = []
        self._running = False
        self._mt5_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        self.is_backtest = is_backtest
        self.leverage = leverage
        
        # Initialize MT5 only if not in backtest mode
        if not is_backtest:
            if not self.initialize_mt5():
                raise Exception("Failed to initialize MT5")
                
            # Log account info once
            account_info = mt5.account_info()
            if account_info is not None:
                self.logger.info(f"MT5 Account Info:")
                self.logger.info(f"Login: {account_info.login}")
                self.logger.info(f"Server: {account_info.server}")
                self.logger.info(f"Balance: {account_info.balance}")
                self.logger.info(f"Equity: {account_info.equity}")
                self.logger.info(f"Margin: {account_info.margin}")
                self.logger.info(f"Free Margin: {account_info.margin_free}")
                self.logger.info(f"Margin Level: {account_info.margin_level}%")
                self.logger.info(f"Leverage: {self.leverage}x")
            else:
                self.logger.error("Failed to get account info")
            
    def initialize_mt5(self):
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize():
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
                
            # Load MT5 account settings from config
            with open('config.json', 'r') as f:
                config = json.load(f)
                
            account = config['mt5_account']['account']
            password = config['mt5_account']['password']
            server = config['mt5_account']['server']
            
            if not mt5.login(account, password=password, server=server):
                self.logger.error(f"MT5 login failed: {mt5.last_error()}")
                return False
                
            self.logger.info("MT5 initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing MT5: {str(e)}")
            return False
            
    def add_strategy(self, strategy):
        """Add a trading strategy to the manager"""
        if isinstance(strategy, BaseTradingStrategy):
            # Set leverage for the strategy
            strategy.leverage = self.leverage
            self.strategies.append(strategy)
            self.logger.info(f"Added strategy: {strategy.__class__.__name__} with {self.leverage}x leverage")
        else:
            raise ValueError("Strategy must be an instance of BaseTradingStrategy")
    
    def _fetch_data(self, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
        """Fetch historical data from MT5."""
        timeframes = {
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        if timeframe not in timeframes:
            self.logger.error(f"Invalid timeframe: {timeframe}")
            return None
            
        if self.is_backtest:
            # In backtest mode, we'll use the strategy's get_data method
            return None
            
        with self._mt5_lock:
            try:
                rates = mt5.copy_rates_from_pos(self.symbol, timeframes[timeframe], 0, bars)
                if rates is None or len(rates) == 0:
                    self.logger.warning(f"No data received for {self.symbol} {timeframe}")
                    return None
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                self.logger.debug(f"Fetched {len(df)} {timeframe} bars for {self.symbol}")
                return df
            except Exception as e:
                self.logger.error(f"Error fetching {timeframe} data: {str(e)}")
                return None
    
    def _strategy_runner(self, strategy: object) -> None:
        """
        Internal method to run a single strategy in a thread.
        
        Args:
            strategy: The strategy instance to run.
        """
        strategy_name = strategy.__class__.__name__
        try:
            while self._running:
                # Check MT5 connection if not in backtest mode
                if not self.is_backtest:
                    with self._mt5_lock:
                        if not mt5.terminal_info():
                            self.logger.warning(f"MT5 disconnected for {strategy_name}. Attempting reconnect...")
                            self._initialize_mt5()
                
                # Fetch data
                if self.is_backtest:
                    # In backtest mode, use strategy's get_data method
                    m15_df = strategy.get_data("M15", 200)
                    h4_df = strategy.get_data("H4", 100)
                    d1_df = strategy.get_data("D1", 100)
                    m5_df = strategy.get_data("M5", 400) if strategy_name == "MultiTimeframeRSIStrategy" else None
                    h1_df = strategy.get_data("H1", 200) if strategy_name == "MultiTimeframeRSIStrategy" else None
                else:
                    # Request more M15 bars for MATradingStrategy to ensure stable EMA calculation
                    # We need at least 1000 bars for reliable EMA signals
                    m15_bars = 1000 if strategy_name == "MATradingStrategy" else 200
                    self.logger.info(f"Requesting {m15_bars} M15 bars for {strategy_name}")
                    m15_df = self._fetch_data("M15", m15_bars)
                    h4_df = self._fetch_data("H4", 100)
                    d1_df = self._fetch_data("D1", 100)
                    m5_df = self._fetch_data("M5", 400) if strategy_name == "MultiTimeframeRSIStrategy" else None
                    h1_df = self._fetch_data("H1", 200) if strategy_name == "MultiTimeframeRSIStrategy" else None
                
                if any(df is None for df in [m15_df, h4_df, d1_df] + ([m5_df, h1_df] if strategy_name == "MultiTimeframeRSIStrategy" else [])):
                    self.logger.warning(f"Skipping cycle for {strategy_name}: Insufficient data")
                    time.sleep(60)
                    continue
                
                # Check signals
                try:
                    signal, signal_type, strength = strategy.check_signals(
                        m15_df,
                        h4_df,
                        d1_df,
                        m5_df if strategy_name == "MultiTimeframeRSIStrategy" else None,
                        h1_df if strategy_name == "MultiTimeframeRSIStrategy" else None
                    )
                    
                    if signal and signal_type in ["buy", "sell"]:
                        # Execute trade
                        if not self.is_backtest:
                            with self._mt5_lock:
                                tick = mt5.symbol_info_tick(self.symbol)
                                if not tick:
                                    self.logger.error(f"Failed to get tick data for {strategy_name}")
                                    time.sleep(60)
                                    continue
                                price = tick.ask if signal_type == "buy" else tick.bid
                        else:
                            # In backtest mode, use the last close price
                            price = m15_df['close'].iloc[-1]
                            
                        entry, sl, tp = strategy.calculate_levels(price, signal_type)
                        
                        # Calculate appropriate volume based on risk per trade (1% of account balance)
                        if not self.is_backtest:
                            account_info = mt5.account_info()
                            if account_info is None:
                                self.logger.error("Failed to get account info")
                                time.sleep(60)
                                continue
                            risk_amount = account_info.balance * 0.01  # 1% risk per trade
                        else:
                            risk_amount = strategy.initial_balance * 0.01  # 1% risk per trade
                            
                        sl_distance = abs(entry - sl) / strategy.point
                        volume = strategy.calculate_volume(risk_amount, sl_distance)
                        
                        # Adjust volume based on leverage
                        volume = volume * (self.leverage / 100)  # Adjust for 2000x leverage
                        
                        order_type = mt5.ORDER_TYPE_BUY if signal_type == "buy" else mt5.ORDER_TYPE_SELL
                        h1_rsi = m15_rsi = m5_rsi = None
                        if strategy_name == "MultiTimeframeRSIStrategy":
                            m5_rsi, _ = strategy.get_rsi_signal("M5", m5_df)
                            m15_rsi, _ = strategy.get_rsi_signal("M15", m15_df)
                            h1_rsi, _ = strategy.get_rsi_signal("H1", h1_df)
                        
                        # Place order and get trade result
                        trade_result = strategy.place_order(
                            order_type=order_type,
                            volume=volume,
                            price=entry,
                            sl=sl,
                            tp=tp,
                            strategy_name=strategy_name,
                            rsi_values=(m5_rsi, m15_rsi, h1_rsi),
                            macd_crossover=False,
                            breakout_distance=strength
                        )
                        
                        # Log trade if successful
                        if trade_result and trade_result.get('ticket'):
                            self.log_trade({
                                'entry_time': pd.Timestamp.now(),
                                'entry_price': entry,
                                'volume': volume,
                                'sl': sl,
                                'tp': tp,
                                'order_type': signal_type,
                                'exit_time': None,
                                'exit_price': None,
                                'profit': 0.0,
                                'status': 'open',
                                'strategy': strategy_name,
                                'priority': 'medium',  # Default priority
                                'leverage': self.leverage  # Add leverage to trade log
                            })
                        
                        self.logger.info(
                            f"Placed {signal_type} order for {strategy_name}: "
                            f"Entry={entry:.5f}, SL={sl:.5f}, TP={tp:.5f}, "
                            f"Volume={volume:.2f}, Leverage={self.leverage}x"
                        )
                        time.sleep(300 if strategy_name == "MultiTimeframeRSIStrategy" else 600)
                    else:
                        time.sleep(60 if strategy_name == "MultiTimeframeRSIStrategy" else 300)
                except Exception as e:
                    self.logger.error(f"Error processing signals for {strategy_name}: {str(e)}")
                    time.sleep(60)
        except Exception as e:
            self.logger.error(f"Fatal error in {strategy_name}: {str(e)}", exc_info=True)
    
    def start(self) -> None:
        """Start all strategies in separate threads."""
        if self._running:
            self.logger.warning("Strategy manager is already running")
            return
        if not self.strategies:
            self.logger.warning("No strategies to run")
            return
        self.logger.info(f"Starting {len(self.strategies)} strategies with {self.leverage}x leverage...")
        self._running = True
        for strategy in self.strategies:
            thread = threading.Thread(
                target=self._strategy_runner,
                args=(strategy,),
                name=f"{strategy.__class__.__name__}_thread"
            )
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
            self.logger.info(f"Started thread for {strategy.__class__.__name__}")
    
    def stop(self) -> None:
        """Stop all strategies and clean up resources."""
        if not self._running:
            self.logger.warning("Strategy manager is not running")
            return
        self.logger.info("Stopping all strategies...")
        self._running = False
        for thread in self.threads:
            thread.join(timeout=5)
        self.threads.clear()
        if not self.is_backtest:
            mt5.shutdown()
        self.logger.info("All strategies stopped")
    
    def run(self) -> None:
        """Run the strategy manager until interrupted."""
        try:
            self.start()
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, stopping...")
        finally:
            self.stop()
    
    def log_trade(self, trade: Dict[str, Any]) -> None:
        """
        Log trade information to CSV file.
        
        Args:
            trade: Dictionary containing trade information
        """
        try:
            # Create trades directory if it doesn't exist
            os.makedirs('trades', exist_ok=True)
            
            # Define CSV file path
            csv_file = os.path.join('trades', f'trades_{self.symbol}.csv')
            
            # Check if file exists to determine if we need to write headers
            file_exists = os.path.isfile(csv_file)
            
            # Define field names
            fieldnames = [
                'entry_time', 'entry_price', 'volume', 'sl', 'tp',
                'order_type', 'exit_time', 'exit_price', 'profit',
                'status', 'strategy', 'priority', 'leverage'
            ]
            
            # Write trade to CSV
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(trade)
                
            self.logger.info(f"Logged trade to {csv_file}")
            
        except Exception as e:
            self.logger.error(f"Error logging trade: {str(e)}")


def main():
    """Main function to run the strategy manager."""
    try:
        # Initialize MT5 first
        if not mt5.initialize():
            logger.error("Failed to initialize MT5")
            return
            
        # Initialize strategy manager with 2000x leverage
        manager = StrategyManager(symbol="XAUUSDm", leverage=2000)
        
        # Add strategies with config path
        manager.add_strategy(RSITradingStrategy(
            symbol="XAUUSDm",
            config_path="config.json",
            mt5_initialized=True
        ))
        manager.add_strategy(MATradingStrategy(
            symbol="XAUUSDm",
            config_path="config.json",
            mt5_initialized=True
        ))
        manager.add_strategy(MultiTimeframeRSIStrategy(
            symbol="XAUUSDm",
            config_path="config.json",
            mt5_initialized=True
        ))
        
        # Run manager
        manager.run()
        
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        if mt5.initialize():
            mt5.shutdown()


if __name__ == "__main__":
    main()