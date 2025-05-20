import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies.multi_timeframe_rsi_strategy import MultiTimeframeRSIStrategy


class TestMultiTimeframeRSIStrategy(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.strategy = MultiTimeframeRSIStrategy(symbol="XAUUSD", config_path="config.json")
        
    def test_get_rsi_signal(self):
        """Test RSI signal generation for different timeframes."""
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
        
        rsi_value, signal = self.strategy.get_rsi_signal("M15", data)
        self.assertIsInstance(rsi_value, float)
        self.assertIn(signal, ['overbought', 'oversold', 'neutral'])
        
    def test_check_signals(self):
        """Test signal generation across multiple timeframes."""
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
        m5_data = m15_data.copy()
        h1_data = m15_data.copy()
        
        signal, signal_type, strength = self.strategy.check_signals(
            m15_data, h4_data, d1_data, m5_data, h1_data
        )
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
        
    def test_calculate_rsi(self):
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
        
        rsi_values = self.strategy.calculate_rsi(data, period=14)
        self.assertIsInstance(rsi_values, pd.Series)
        self.assertFalse(rsi_values.empty)
        
    def test_generate_mock_data(self):
        """Test mock data generation."""
        timeframe = "M15"
        bars = 100
        
        data = self.strategy._generate_mock_data(timeframe, bars)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), bars)
        self.assertIn('time', data.columns)
        self.assertIn('open', data.columns)
        self.assertIn('high', data.columns)
        self.assertIn('low', data.columns)
        self.assertIn('close', data.columns)
        self.assertIn('tick_volume', data.columns)


if __name__ == '__main__':
    unittest.main() 