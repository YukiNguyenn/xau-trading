"""
Strategy Manager

This module provides a manager class to run multiple trading strategies in parallel.
"""

import logging
import threading
import time
from typing import List, Dict, Any, Optional
import MetaTrader5 as mt5
from strategies.rsi_trading_strategy import RSITradingStrategy
from strategies.ma_trading_strategy import MATradingStrategy
from strategies.multi_timeframe_rsi_strategy import MultiTimeframeRSIStrategy

class StrategyManager:
    """
    Manages and runs multiple trading strategies in parallel using threading.
    """
    
    def __init__(
        self,
        symbol: str,
        config_path: str,
        initial_balance: float = 100.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the strategy manager.
        
        Args:
            symbol: Trading symbol
            config_path: Path to configuration file
            initial_balance: Initial balance for each strategy
            logger: Optional logger for logging events
        """
        self.symbol = symbol
        self.config_path = config_path
        self.initial_balance = initial_balance
        
        # Initialize logger
        if logger is None:
            self.logger = self._setup_logger()
        else:
            self.logger = logger
            
        # Initialize MT5 once for all strategies
        if not self.initialize_mt5():
            raise Exception("MT5 initialization failed")
            
        # Initialize strategies
        self.strategies = self._initialize_strategies()
        
        # Thread management
        self.threads: Dict[str, threading.Thread] = {}
        self.running = False
        
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the strategy manager."""
        logger = logging.getLogger("StrategyManager")
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
            
    def _initialize_strategies(self) -> Dict[str, Any]:
        """Initialize all trading strategies."""
        strategies = {}
        
        # Initialize RSI Strategy
        strategies['RSI'] = RSITradingStrategy(
            symbol=self.symbol,
            config_path=self.config_path,
            mt5_initialized=True,
            logger=self.logger,
            initial_balance=self.initial_balance
        )
        
        # Initialize MA Strategy
        strategies['MA'] = MATradingStrategy(
            symbol=self.symbol,
            config_path=self.config_path,
            mt5_initialized=True,
            logger=self.logger,
            initial_balance=self.initial_balance
        )
        
        # Initialize Multi-Timeframe RSI Strategy
        strategies['MultiTimeframeRSI'] = MultiTimeframeRSIStrategy(
            symbol=self.symbol,
            config_path=self.config_path,
            mt5_initialized=True,
            logger=self.logger,
            initial_balance=self.initial_balance
        )
        
        return strategies
        
    def _run_strategy(self, strategy_name: str, strategy: Any):
        """Run a single strategy in a separate thread."""
        try:
            self.logger.info(f"Starting {strategy_name} strategy")
            strategy.run_strategy()
        except Exception as e:
            self.logger.error(f"Error in {strategy_name} strategy: {str(e)}")
            
    def start(self):
        """Start all strategies in parallel."""
        if self.running:
            self.logger.warning("Strategies are already running")
            return
            
        self.running = True
        self.logger.info("Starting all strategies")
        
        # Start each strategy in a separate thread
        for name, strategy in self.strategies.items():
            thread = threading.Thread(
                target=self._run_strategy,
                args=(name, strategy),
                name=f"Thread-{name}"
            )
            thread.daemon = True  # Thread will be killed when main program exits
            self.threads[name] = thread
            thread.start()
            self.logger.info(f"Started {name} strategy thread")
            
    def stop(self):
        """Stop all running strategies."""
        if not self.running:
            self.logger.warning("No strategies are running")
            return
            
        self.running = False
        self.logger.info("Stopping all strategies")
        
        # Wait for all threads to complete
        for name, thread in self.threads.items():
            thread.join(timeout=5)  # Wait up to 5 seconds for thread to finish
            self.logger.info(f"Stopped {name} strategy thread")
            
        # Shutdown MT5
        mt5.shutdown()
        self.logger.info("MT5 connection closed")
        
    def get_strategy_status(self) -> Dict[str, bool]:
        """Get the running status of each strategy."""
        return {
            name: thread.is_alive()
            for name, thread in self.threads.items()
        }
        
def main():
    """Test the strategy manager."""
    try:
        # Initialize strategy manager
        manager = StrategyManager(
            symbol="XAUUSD",
            config_path="backtest/config/backtest_config.json",
            initial_balance=100.0
        )
        
        # Start all strategies
        manager.start()
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
                status = manager.get_strategy_status()
                print("\nStrategy Status:")
                for name, is_running in status.items():
                    print(f"{name}: {'Running' if is_running else 'Stopped'}")
        except KeyboardInterrupt:
            print("\nStopping strategies...")
            manager.stop()
            
    except Exception as e:
        print(f"Error: {str(e)}")
        
if __name__ == "__main__":
    main() 