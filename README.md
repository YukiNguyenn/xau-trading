# XAUUSD Trading Bot

Một chiến lược giao dịch tự động cho vàng (XAUUSD) sử dụng MetaTrader 5 API với các chỉ báo kỹ thuật RSI và MACD.

## Cấu trúc thư mục

```
xau-trading-bot/
├── strategy/               # Module chiến lược
│   ├── constants.py       # Cấu hình và hằng số
│   ├── indicators.py      # Chỉ báo kỹ thuật (RSI, MACD)
│   ├── backtest.py        # Logic backtest
│   ├── trade_manager.py   # Quản lý giao dịch
│   └── utils.py           # Các hàm tiện ích
├── config.json            # Cấu hình chính
├── CONFIGURATION.md       # Hướng dẫn cấu hình chi tiết
├── requirements.txt       # Yêu cầu thư viện
├── rsi_trading_strategy.py # File chính chạy chiến lược
├── README.md             # Hướng dẫn sử dụng
└── trade_log.csv         # File log giao dịch
```

## Yêu cầu hệ thống

- Python 3.8+
- MetaTrader 5 Terminal (cài đặt và chạy)
- MetaTrader 5 API
- pandas
- numpy
- pandas_ta

## Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

2. Cấu hình tài khoản MT5:
   - Mở file `config.json`
   - Cập nhật thông tin tài khoản MT5:
   ```json
   "mt5_account": {
       "account": "YOUR_ACCOUNT_NUMBER",
       "password": "YOUR_PASSWORD",
       "server": "YOUR_SERVER_NAME"
   }
   ```
   - Thay YOUR_ACCOUNT_NUMBER, YOUR_PASSWORD và YOUR_SERVER_NAME bằng thông tin tài khoản MT5 thực tế của bạn

3. Đảm bảo MetaTrader 5 đang chạy và kết nối với tài khoản của bạn.

## Cấu hình

Các cấu hình chính của bot được định nghĩa trong file `config.json`. Dưới đây là giải thích chi tiết về từng cấu hình:

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

1. Chạy chiến lược giao dịch:
```bash
python rsi_trading_strategy.py
```

2. Kiểm tra kết quả backtest:
```python
from rsi_trading_strategy import AdvancedTradingStrategy

# Tạo đối tượng chiến lược
strategy = AdvancedTradingStrategy(symbol="XAUUSD")

# Kiểm tra trong 3 tháng gần đây
results = strategy.backtest(
    datetime.now() - timedelta(days=90),
    datetime.now()
)

# Hiển thị kết quả
print(f"Win rate: {results['win_rate']:.2%}")
print(f"Profit factor: {results['profit_factor']:.2f}")
print(f"Max drawdown: ${results['max_drawdown']:.2f}")
print(f"Final equity: ${results['final_equity']:.2f}")
print(f"Average duration: {results['avg_duration']:.1f} H4 candles")
```

### Sử dụng Docker

#### Sử dụng Docker Compose (Đề xuất)

1. Build và chạy container:
```bash
docker-compose up -d
```

2. Kiểm tra log:
```bash
docker-compose logs -f
```

3. Dừng và xóa container:
```bash
docker-compose down
```

#### Sử dụng Docker đơn lẻ

1. Build Docker image:
```bash
docker build -t xau-trading-bot .
```

2. Chạy container:
```bash
docker run -d --name trading-bot xau-trading-bot
```

3. Kiểm tra log:
```bash
docker logs -f trading-bot
```

4. Dừng container:
```bash
docker stop trading-bot
```

5. Xóa container:
```bash
docker rm trading-bot
```

**Lưu ý:** Khi sử dụng Docker, cần đảm bảo MetaTrader 5 Terminal đang chạy trên cùng máy chủ và cấu hình để cho phép kết nối từ container Docker.

## Chiến lược giao dịch

Chiến lược sử dụng các yếu tố sau:

1. **Xác định xu hướng**:
   - Sử dụng khung thời gian D1 để xác định xu hướng chính
   - Kiểm tra breakout khỏi vùng supply/demand trên H4

2. **Chỉ báo kỹ thuật**:
   - RSI (3 khung thời gian: 6, 14, 24)
   - MACD crossover
   - Kiểm tra tín hiệu RSI và MACD crossover

3. **Quản lý rủi ro**:
   - Stop loss: 500 points ($5)
   - Take profit: 1000 points ($10)
   - Trailing stop: 300 points
   - Kích thước lô: 0.01 lot

4. **Thời gian giao dịch**:
   - Kiểm tra tín hiệu mỗi 15 phút
   - Thời gian tối đa cho mỗi lệnh: 1 tuần (24 nến H4)
   - Giới hạn 50 nến để kiểm tra tín hiệu breakout

## Lưu ý

1. Đảm bảo tài khoản MetaTrader 5 có đủ số dư để giao dịch
2. Kiểm tra spread và commission của broker
3. Sử dụng tài khoản demo trước khi giao dịch thật
4. Theo dõi log giao dịch trong file `trade_log.csv`

## Liên hệ

Nếu có bất kỳ câu hỏi hoặc vấn đề gì, vui lòng liên hệ thông qua email hoặc tạo issue trên GitHub.
