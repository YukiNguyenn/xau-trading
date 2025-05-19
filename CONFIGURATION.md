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
