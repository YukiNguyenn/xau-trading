"""Các hằng số và cấu hình cho chiến lược giao dịch"""

# Khung thời gian
timeframes = {
    'W1': mt5.TIMEFRAME_W1,
    'D1': mt5.TIMEFRAME_D1,
    'H4': mt5.TIMEFRAME_H4,
    'M15': mt5.TIMEFRAME_M15
}

# Cài đặt quản lý rủi ro
risk_params = {
    'stop_loss_points': 500,  # 500 points ($5)
    'take_profit_points': 1000,  # 1000 points ($10)
    'trailing_stop_points': 300,  # Di chuyển SL sau 300 points
    'price_threshold': 0.001,  # 0.1% biến động giá
    'zone_threshold': 0.005,  # ±0.5% cho vùng supply/demand
}

# Cài đặt RSI
rsi_params = {
    'short_period': 6,
    'medium_period': 14,
    'long_period': 24,
    
    'short_overbought': 80,
    'short_oversold': 20,
    'medium_overbought': 70,
    'medium_oversold': 30,
    'long_overbought': 60,
    'long_oversold': 40,
}

# Cài đặt MACD
macd_params = {
    'fast': 12,
    'slow': 26,
    'signal': 9,
}

# Cài đặt spread và commission
trading_params = {
    'spread_points': 20,  # Spread trung bình
    'commission': 0.0001,  # 0.01% commission
    'max_duration_candles': 50,  # Số nến tối đa để kiểm tra
    'max_week_duration': 24,  # 24 nến H4 = 1 tuần
}
