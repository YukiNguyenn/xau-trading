import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies.rsi_trading_strategy import RSITradingStrategy


class TestRSITradingStrategy(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.strategy = RSITradingStrategy(symbol="XAUUSD", config_path="config.json")
        
    def test_calculate_trend(self):
        """Test trend calculation."""
        # Create test data
        dates = pd.date_range(start='2024-01-01', periods=200, freq='15min')
        data = pd.DataFrame({
            'time': dates,
            'open': np.random.normal(2000, 10, 200),
            'high': np.random.normal(2005, 10, 200),
            'low': np.random.normal(1995, 10, 200),
            'close': np.random.normal(2000, 10, 200),
            'tick_volume': np.random.randint(100, 1000, 200)
        })
        
        # Mock the get_data method
        def mock_get_data(timeframe, bars=100):
            return data
            
        self.strategy.get_data = mock_get_data
        
        trend = self.strategy.calculate_trend('M15')
        self.assertIn(trend, ['bullish', 'bearish', 'neutral'])
        
    def test_check_signals(self):
        """Test signal generation."""
        # Create test data
        dates = pd.date_range(start='2024-01-01', periods=200, freq='15min')
        m15_data = pd.DataFrame({
            'time': dates,
            'open': np.random.normal(2000, 10, 200),
            'high': np.random.normal(2005, 10, 200),
            'low': np.random.normal(1995, 10, 200),
            'close': np.random.normal(2000, 10, 200),
            'tick_volume': np.random.randint(100, 1000, 200)
        })
        
        h4_data = m15_data.copy()
        d1_data = m15_data.copy()
        
        signal, signal_type, strength = self.strategy.check_signals(m15_data, h4_data, d1_data)
        self.assertIsInstance(signal, bool)
        self.assertIsInstance(signal_type, (str, type(None)))
        self.assertIsInstance(strength, float)
        
    def test_calculate_levels(self):
        """Test level calculations."""
        price = 2000.0
        trend = 'bullish'
        
        entry, sl, tp = self.strategy.calculate_levels(price, trend)
        self.assertIsInstance(entry, float)
        self.assertIsInstance(sl, float)
        self.assertIsInstance(tp, float)
        self.assertLess(sl, entry)
        self.assertGreater(tp, entry)
        
    def test_calculate_rsi_indicators(self):
        """Test RSI calculation."""
        # Create test data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
        data = pd.DataFrame({
            'time': dates,
            'open': np.random.normal(2000, 10, 100),
            'high': np.random.normal(2005, 10, 100),
            'low': np.random.normal(1995, 10, 100),
            'close': np.random.normal(2000, 10, 100),
            'tick_volume': np.random.randint(100, 1000, 100)
        })
        
        rsi_short, rsi_medium, rsi_long = self.strategy.calculate_rsi_indicators(data)
        self.assertIsInstance(rsi_short, (float, type(None)))
        self.assertIsInstance(rsi_medium, (float, type(None)))
        self.assertIsInstance(rsi_long, (float, type(None)))
        
    def test_check_rsi_signals(self):
        """Test RSI signal generation."""
        rsi_short = 30
        rsi_medium = 35
        rsi_long = 40
        trend = 'bullish'
        
        signal, signal_type = self.strategy.check_rsi_signals(rsi_short, rsi_medium, rsi_long, trend)
        self.assertIsInstance(signal, bool)
        self.assertIsInstance(signal_type, (str, type(None)))


if __name__ == '__main__':
    unittest.main() 