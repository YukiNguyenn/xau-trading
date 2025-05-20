# Configuration Guide

This document provides detailed information about the configuration settings for the XAU Trading Bot.

## Initial Balance

The default initial balance for backtesting is set to $100. This affects various risk management parameters:

- Max Daily Loss: $10 (10% of initial balance)
- Max Drawdown: 10% of initial balance
- Position sizing is calculated based on this initial balance

## Trading Parameters

### Volume Settings
```json
"trading": {
    "min_volume": 0.01,    // Minimum trading volume
    "max_volume": 0.1,     // Maximum trading volume
    "max_open_positions": 4 // Maximum number of open positions
}
```

### Risk Management
```json
"risk": {
    "stop_loss_points": 100,      // Stop loss in points
    "take_profit_points": 200,    // Take profit in points
    "trailing_stop_points": 50,   // Trailing stop in points
    "max_daily_loss": 10,         // Maximum daily loss ($)
    "max_drawdown": 0.1           // Maximum drawdown (10%)
}
```

## Technical Indicators

### Moving Averages (EMA)
```json
"ma": {
    "fast": 34,    // Fast EMA period
    "medium": 89,  // Medium EMA period
    "slow": 200,   // Slow EMA period
    "type": "ema"  // Indicator type
}
```

Minimum bars required for EMA calculation:
- EMA 34: Minimum 34 bars
- EMA 89: Minimum 89 bars
- EMA 200: Minimum 200 bars
- Recommended: At least 1000 bars for stable signals

### RSI Settings
```json
"rsi": {
    "periods": {
        "short": 6,   // Short-term RSI
        "medium": 14, // Medium-term RSI
        "long": 24    // Long-term RSI
    },
    "thresholds": {
        "short": {
            "overbought": 80,
            "oversold": 20
        },
        "medium": {
            "overbought": 75,
            "oversold": 25
        },
        "long": {
            "overbought": 70,
            "oversold": 30
        }
    }
}
```

### MACD Settings
```json
"macd": {
    "fast": 12,    // Fast period
    "slow": 26,    // Slow period
    "signal": 9    // Signal period
}
```

## Position Management

### Priority Levels
```json
"priority_levels": {
    "high": {
        "rsi_short": 70,
        "rsi_medium": 70,
        "rsi_long": 70,
        "macd_crossover": true
    },
    "medium": {
        "rsi_short": 60,
        "rsi_medium": 60,
        "rsi_long": 60,
        "macd_crossover": true
    }
}
```

## Timeframes

Available timeframes and their durations (in minutes):
```json
"timeframes": {
    "M5": 5,    // 5 minutes
    "M15": 15,  // 15 minutes
    "H1": 60,   // 1 hour
    "H4": 240,  // 4 hours
    "D1": 1440  // 1 day
}
```

## Trading Costs

```json
"costs": {
    "spread": {
        "max_points": 5  // Maximum spread in points
    }
},
"trading": {
    "commission": 0.0001,  // Commission per trade
    "spread_points": 5     // Spread in points
}
```

## Logging Configuration

```json
"logging": {
    "level": "INFO",           // Logging level
    "file": "trading_bot.log", // Log file name
    "max_size": 10485760,      // Maximum log file size (10MB)
    "backup_count": 5          // Number of backup files
}
```

## Important Notes

1. **Data Requirements**:
   - The bot automatically fetches sufficient historical data for calculations
   - For EMA strategies, at least 1000 M15 bars are recommended
   - Data is cached in the `data` directory for faster access

2. **Risk Management**:
   - All risk parameters are calculated based on the initial balance
   - Position sizing is automatically adjusted based on account balance
   - Stop loss and take profit levels are calculated in points

3. **Performance Considerations**:
   - Higher timeframe data requires more historical bars
   - More indicators increase calculation time
   - Consider system resources when adjusting parameters

4. **Backtesting vs Live Trading**:
   - Backtest configuration uses more lenient position limits
   - Live trading has stricter risk management
   - Some parameters may differ between backtest and live modes

# Cấu hình chiến lược giao dịch

## Cấu hình chính

Các cấu hình chính của bot được định nghĩa trong file `config.json`. Dưới đây là giải thích chi tiết về từng cấu hình:

### Cấu hình Tài khoản MT5

```json
"mt5_account": {
    "account": "YOUR_ACCOUNT_NUMBER",
    "password": "YOUR_PASSWORD",
    "server": "YOUR_SERVER_NAME"
}
```

- `account`: Số tài khoản MT5 của bạn
- `password`: Mật khẩu tài khoản MT5
- `server`: Tên server MT5 (ví dụ: "ICMarketsSC-Demo" hoặc "ICMarketsSC-Live")

**Lưu ý:** Đảm bảo thông tin tài khoản MT5 chính xác để bot có thể đăng nhập thành công. Nếu thông tin không chính xác, bot sẽ không thể thực hiện các lệnh giao dịch.

### Cấu hình Giao dịch (`trading`)

- `symbol`: Biểu đồ giao dịch (XAUUSD)
- `stop_loss_points`: Điểm dừng lỗ (500 points ≈ $5)
- `take_profit_points`: Điểm chốt lời (1000 points ≈ $10)
- `trailing_stop_points`: Di chuyển điểm dừng lỗ sau 300 points
- `min_volume`: Khối lượng giao dịch tối thiểu (0.01 lot)
- `max_volume`: Khối lượng giao dịch tối đa (1 lot)
- `price_threshold`: Ngưỡng biến động giá (0.1%)
- `zone_threshold`: Ngưỡng vùng supply/demand (±0.5%)

### Cấu hình Chỉ báo Kỹ thuật (`indicators`)

#### RSI (Relative Strength Index)
- `periods`:
  - `short`: 6 kỳ (RSI ngắn hạn)
  - `medium`: 14 kỳ (RSI trung hạn)
  - `long`: 24 kỳ (RSI dài hạn)

- `thresholds`:
  - `overbought`: Ngưỡng quá mua
  - `oversold`: Ngưỡng quá bán

  Mỗi khung thời gian có các ngưỡng riêng:
  - Ngắn hạn: quá mua 80, quá bán 20
  - Trung hạn: quá mua 70, quá bán 30
  - Dài hạn: quá mua 60, quá bán 40

#### MACD (Moving Average Convergence Divergence)
- `fast`: 12 kỳ (đường nhanh)
- `slow`: 26 kỳ (đường chậm)
- `signal`: 9 kỳ (đường tín hiệu)

### Cấu hình Chi phí (`costs`)
- `spread_points`: Spread trung bình (20 points)
- `commission`: Phí giao dịch (0.01%)

## Cách sử dụng

### Sử dụng trực tiếp
