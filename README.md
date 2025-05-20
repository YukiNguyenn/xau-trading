# XAU Trading Bot

A multi-strategy trading bot for XAUUSD (Gold) using MetaTrader 5.

## Project Structure

```
xau-trading-bot/
├── config.json                 # Configuration file
├── strategy_manager.py         # Main strategy manager
├── results/                    # Trading results and logs
│   ├── rsitradingstrategy_trade_log.csv
│   ├── matradingstrategy_trade_log.csv
│   └── multitimeframersistrategy_trade_log.csv
├── strategies/                 # Trading strategies
│   ├── core/                  # Core components
│   │   ├── base_trading_strategy.py
│   │   ├── data_manager.py
│   │   ├── indicators.py
│   │   ├── risk_manager.py
│   │   ├── trade_manager.py
│   │   └── utils.py
│   ├── rsi_trading_strategy.py
│   ├── ma_trading_strategy.py
│   └── multi_timeframe_rsi_strategy.py
└── tests/                      # Test files
    ├── test_rsi_strategy.py
    ├── test_ma_strategy.py
    └── test_multi_timeframe_strategy.py
```

## Features

- Multiple trading strategies:
  - RSI Trading Strategy
  - Moving Average (MA) Trading Strategy
  - Multi-Timeframe RSI Strategy
- Real-time market data processing
- Risk management
- Trade logging and analysis
- Configurable parameters
- Multi-threaded execution

## Requirements

- Python 3.8+
- MetaTrader 5
- Required Python packages (see requirements.txt):
  - pandas
  - numpy
  - MetaTrader5
  - pandas-ta

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/xau-trading-bot.git
cd xau-trading-bot
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Configure MetaTrader 5:
   - Install MetaTrader 5
   - Create a demo account
   - Update config.json with your MT5 credentials

## Configuration

Edit `config.json` to customize:
- MT5 account settings
- Trading parameters
- Strategy-specific settings
- Risk management rules

Example configuration:
```json
{
    "mt5_account": {
        "account": "your_account_number",
        "password": "your_password",
        "server": "your_broker_server"
    },
    "trading": {
        "min_volume": 0.01,
        "max_volume": 1.0,
        "stop_loss_points": 100,
        "take_profit_points": 200
    },
    "indicators": {
        "rsi": {
            "periods": {
                "short": 14,
                "medium": 21,
                "long": 50
            },
            "thresholds": {
                "short": {
                    "overbought": 70,
                    "oversold": 30
                },
                "medium": {
                    "overbought": 65,
                    "oversold": 35
                },
                "long": {
                    "overbought": 60,
                    "oversold": 40
                }
            }
        },
        "ma": {
            "fast": 10,
            "slow": 50
        }
    }
}
```

## Usage

1. Start the trading bot:
```bash
python strategy_manager.py
```

2. Monitor trading activity:
- Check the logs in the `results` directory
- Monitor MT5 terminal for open positions
- Review trade logs for performance analysis

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred through the use of this software.
