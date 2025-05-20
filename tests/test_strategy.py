import pytest
from strategies import AdvancedTradingStrategy
import mock
import mt5

def test_calculate_priority():
    """Kiểm tra tính toán priority dựa trên các chỉ báo"""
    strategy = AdvancedTradingStrategy()
    
    # Test case 1: RSI mạnh, MACD crossover, breakout lớn
    priority = strategy._get_priority(
        rsi_short=85,
        rsi_medium=75,
        rsi_long=65,
        macd_crossover=True,
        breakout_distance=0.015
    )
    assert priority == 'high'
    
    # Test case 2: RSI trung bình, MACD crossover, breakout vừa phải
    priority = strategy._get_priority(
        rsi_short=75,
        rsi_medium=65,
        rsi_long=55,
        macd_crossover=True,
        breakout_distance=0.01
    )
    assert priority == 'medium'
    
    # Test case 3: RSI yếu, MACD không crossover, breakout nhỏ
    priority = strategy._get_priority(
        rsi_short=70,
        rsi_medium=60,
        rsi_long=50,
        macd_crossover=False,
        breakout_distance=0.005
    )
    assert priority == 'low'
    
    # Test case 4: RSI yếu, MACD không crossover, breakout rất nhỏ
    priority = strategy._get_priority(
        rsi_short=70,
        rsi_medium=60,
        rsi_long=50,
        macd_crossover=False,
        breakout_distance=0.001
    )
    assert priority == 'low'
    
    # Test case 5: RSI vừa phải, MACD crossover, breakout lớn
    priority = strategy._get_priority(
        rsi_short=75,
        rsi_medium=65,
        rsi_long=55,
        macd_crossover=True,
        breakout_distance=0.02
    )
    assert priority == 'high'

def test_close_low_priority_position():
    """Kiểm tra việc đóng vị thế có ưu tiên thấp hơn"""
    strategy = AdvancedTradingStrategy()
    
    # Mock MT5 functions
    mock_position = mock.Mock()
    mock_position.volume = 0.01
    mock_position.type = mt5.ORDER_TYPE_BUY
    mock_position.ticket = 12345
    
    mock_symbol_info = mock.Mock()
    mock_symbol_info.ask = 1900.0
    mock_symbol_info.bid = 1899.0
    
    mock_positions_get = mock.Mock(return_value=[mock_position])
    mock_symbol_info_tick = mock.Mock(return_value=mock_symbol_info)
    mock_order_send = mock.Mock(return_value=mock.Mock(retcode=mt5.TRADE_RETCODE_DONE))
    
    # Mock MT5 module
    with mock.patch('mt5.positions_get', mock_positions_get), \
         mock.patch('mt5.symbol_info_tick', mock_symbol_info_tick), \
         mock.patch('mt5.order_send', mock_order_send):
        
        # Thêm vị thế vào danh sách
        strategy.open_trades = [{
            'ticket': 12345,
            'entry': 1900.0,
            'sl': 1850.0,
            'tp': 1950.0,
            'priority': 'low',
            'profit': 0
        }]
        
        # Kiểm tra đóng vị thế
        strategy._close_low_priority_trades('high')
        
        # Kiểm tra vị thế đã được đóng
        assert len(strategy.open_trades) == 0
        
        # Kiểm tra MT5 functions được gọi đúng
        mock_positions_get.assert_called_once_with(ticket=12345)
        mock_symbol_info_tick.assert_called_once_with(strategy.symbol)
        mock_order_send.assert_called_once()

def test_close_position_error_handling():
    """Kiểm tra xử lý lỗi khi đóng vị thế"""
    strategy = AdvancedTradingStrategy()
    
    # Test case 1: Vị thế không tồn tại
    assert not strategy.close_position(99999)
    
    # Test case 2: MT5 không khởi động
    strategy.initialize_mt5 = mock.Mock(return_value=False)
    assert not strategy.close_position(12345)
    
    # Test case 3: Không thể lấy thông tin vị thế
    mock_positions_get = mock.Mock(return_value=None)
    with mock.patch('mt5.positions_get', mock_positions_get):
        assert not strategy.close_position(12345)
    
    # Test case 4: Không thể lấy giá hiện tại
    mock_positions_get = mock.Mock(return_value=[mock.Mock()])
    mock_symbol_info_tick = mock.Mock(return_value=None)
    with mock.patch('mt5.positions_get', mock_positions_get), \
         mock.patch('mt5.symbol_info_tick', mock_symbol_info_tick):
        assert not strategy.close_position(12345)

if __name__ == "__main__":
    pytest.main(["-v", "test_strategy.py"])
