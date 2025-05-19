# Cấu hình chiến lược giao dịch

## Cài đặt giao dịch (Trading Settings)

### Thông tin chung
- `symbol`: Biểu đồ giao dịch (mặc định: XAUUSD)

### Quản lý rủi ro (Risk Management)
- `stop_loss_points`: Điểm dừng lỗ (500 points ≈ $5)
- `take_profit_points`: Điểm chốt lời (1000 points ≈ $10)
- `trailing_stop_points`: Di chuyển điểm dừng lỗ sau 300 points

### Khối lượng giao dịch (Volume Settings)
- `min_volume`: Khối lượng giao dịch tối thiểu (0.01 lot)
- `max_volume`: Khối lượng giao dịch tối đa (1 lot)

### Ngưỡng biến động (Thresholds)
- `price_threshold`: Ngưỡng biến động giá (0.1%)
- `zone_threshold`: Ngưỡng vùng supply/demand (±0.5%)

## Chỉ báo kỹ thuật (Technical Indicators)

### RSI (Relative Strength Index)
- `periods`: Chu kỳ tính toán RSI
  - `short`: 6 kỳ (RSI ngắn hạn)
  - `medium`: 14 kỳ (RSI trung hạn)
  - `long`: 24 kỳ (RSI dài hạn)

- `thresholds`: Ngưỡng quá mua/quá bán
  - `short`: Ngưỡng RSI ngắn hạn (80/20)
  - `medium`: Ngưỡng RSI trung hạn (70/30)
  - `long`: Ngưỡng RSI dài hạn (60/40)

### MACD (Moving Average Convergence Divergence)
- `fast`: Đường nhanh (12 kỳ)
- `slow`: Đường chậm (26 kỳ)
- `signal`: Đường tín hiệu (9 kỳ)

## Chi phí giao dịch (Trading Costs)

- `spread_points`: Spread trung bình (20 points)
- `commission`: Phí giao dịch (0.01%)
