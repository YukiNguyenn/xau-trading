import MetaTrader5 as mt5
from datetime import datetime, timedelta

mt5.initialize()
symbol = "XAUUSDm"
end_date = datetime.now()
start_date = end_date - timedelta(days=348)
rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_date, end_date)
print(f"Retrieved {len(rates) if rates is not None else 'None'} M5 bars for {symbol} from {start_date} to {end_date}")
mt5.shutdown()