import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
from pandas_ta import rsi, macd

import json
import os

class AdvancedTradingStrategy:
    def __init__(self, symbol=None):
        # Load configuration from file
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Trading settings
        self.symbol = symbol if symbol else config['trading']['symbol']
        self.timeframes = {
            'W1': mt5.TIMEFRAME_W1,
            'D1': mt5.TIMEFRAME_D1,
            'H4': mt5.TIMEFRAME_H4,
            'M15': mt5.TIMEFRAME_M15
        }
        
        # Risk management
        self.stop_loss_points = self.config['trading']['stop_loss_points']
        self.take_profit_points = self.config['trading']['take_profit_points']
        self.trailing_stop_points = self.config['trading']['trailing_stop_points']
        self.min_volume = self.config['trading']['min_volume']
        self.max_volume = self.config['trading']['max_volume']
        self.price_threshold = self.config['trading']['price_threshold']
        self.zone_threshold = self.config['trading']['zone_threshold']
        
        # RSI settings
        rsi_config = self.config['indicators']['rsi']
        self.rsi_short = rsi_config['periods']['short']
        self.rsi_medium = rsi_config['periods']['medium']
        self.rsi_long = rsi_config['periods']['long']
        self.rsi_short_overbought = rsi_config['thresholds']['short']['overbought']
        self.rsi_short_oversold = rsi_config['thresholds']['short']['oversold']
        self.rsi_medium_overbought = rsi_config['thresholds']['medium']['overbought']
        self.rsi_medium_oversold = rsi_config['thresholds']['medium']['oversold']
        self.rsi_long_overbought = rsi_config['thresholds']['long']['overbought']
        self.rsi_long_oversold = rsi_config['thresholds']['long']['oversold']
        
        # MACD settings
        macd_config = self.config['indicators']['macd']
        self.macd_fast = macd_config['fast']
        self.macd_slow = macd_config['slow']
        self.macd_signal = macd_config['signal']
        
        # Cost settings
        self.spread_points = self.config['costs']['spread_points']
        self.commission = self.config['costs']['commission']
        
        # Position management settings
        position_config = self.config['position_management']
        self.max_open_positions = position_config['max_open_positions']
        self.priority_levels = position_config['priority_levels']
        
        # MT5 account settings
        self.account = self.config['mt5_account']['account']
        self.password = self.config['mt5_account']['password']
        self.server = self.config['mt5_account']['server']
        
        # Khởi tạo kết nối MT5
        if not self.initialize_mt5():
            quit()
        print("MT5 initialized and logged in successfully")

        # Tạo file log nếu chưa tồn tại
        try:
            with open('trade_log.csv', 'r') as f:
                pass
        except FileNotFoundError:
            with open('trade_log.csv', 'w') as f:
                f.write("time,ticket,entry,sl,tp,result\n")

        # Danh sách ticket đang mở
        self.open_tickets = set()
        
        # Danh sách các vị thế đang mở
        self.open_trades = []

    def _get_priority(self, rsi_short, rsi_medium, rsi_long, macd_crossover):
        """Xác định cấp độ ưu tiên của lệnh dựa trên các chỉ báo"""
        for level in ['high', 'medium', 'low']:
            level_config = self.priority_levels[level]
            if (rsi_short <= level_config['rsi_short'] and
                rsi_medium <= level_config['rsi_medium'] and
                rsi_long <= level_config['rsi_long'] and
                (level_config['macd_crossover'] or macd_crossover)):
                return level
        return 'low'

    def _close_low_priority_trades(self, new_priority):
        """Đóng các vị thế có ưu tiên thấp hơn để nhường chỗ cho lệnh mới"""
        for trade in self.open_trades[:]:  # Copy list để tránh lỗi khi thay đổi trong vòng lặp
            if trade['priority'] < new_priority:
                self.close_position(trade['ticket'])
                self.open_trades.remove(trade)

    def close_position(self, ticket):
        """Đóng một vị thế cụ thể"""
        try:
            position = mt5.positions_get(ticket=ticket)
            if position:
                position = position[0]
                
                # Tạo lệnh đóng vị thế
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": position.volume,
                    "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "price": mt5.symbol_info_tick(self.symbol).ask if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).bid,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                    "position": ticket
                }
                
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"Position {ticket} closed successfully")
                    self.open_tickets.remove(ticket)
                    return True
                else:
                    print(f"Failed to close position {ticket}: {mt5.last_error()}")
                    return False
            return False
            
        except Exception as e:
            print(f"Error closing position {ticket}: {str(e)}")
            return False

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    def get_data(self, timeframe, bars):
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframes[timeframe], 0, bars)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
            if len(df) == 0:
                raise ValueError(f"No data received for {self.symbol} {timeframe}")
                
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception as e:
            print(f"Error in get_data: {str(e)}")
            raise ValueError(f"Failed to get data: {str(e)}")

    def calculate_trend(self, timeframe):
        """
        Tính toán xu hướng dựa trên EMA50 và EMA200 với xử lý lỗi
        Returns:
            str: 'bullish', 'bearish', hoặc 'neutral'
            str: Thông báo lỗi (nếu có)
        """
        try:
            df = self.get_data(timeframe, 200)
            if df is None or len(df) < 200:
                raise ValueError(f"Not enough data for {timeframe} timeframe")
                
            # Tính EMA
            df['ema50'] = df['close'].ewm(span=50).mean()
            df['ema200'] = df['close'].ewm(span=200).mean()
            
            if 'ema50' not in df.columns or 'ema200' not in df.columns:
                raise ValueError("Failed to calculate EMAs")
                
            last_row = df.iloc[-1]
            
            if last_row['ema50'] > last_row['ema200']:
                return 'bullish', ""
            elif last_row['ema50'] < last_row['ema200']:
                return 'bearish', ""
            else:
                return 'neutral', ""
                
        except Exception as e:
            error_msg = str(e)
            print(f"Error in calculate_trend: {error_msg}")
            return 'neutral', error_msg

    def identify_zones(self, timeframe):
        """
        Xác định vùng supply/demand trên khung H4 với xử lý lỗi
        Returns:
            list: Danh sách các vùng supply/demand
            str: Thông báo lỗi (nếu có)
        """
        try:
            df = self.get_data(timeframe, 100)
            if df is None or len(df) < 100:
                raise ValueError(f"Not enough data for {timeframe} timeframe")
                
            zones = []
            for i in range(2, len(df) - 2):
                current = df.iloc[i]
                
                # Kiểm tra vùng supply
                if (current['high'] > df.iloc[i-2]['high'] and 
                    current['high'] > df.iloc[i-1]['high'] and 
                    current['high'] > df.iloc[i+1]['high'] and 
                    current['high'] > df.iloc[i+2]['high']):
                    
                    # Kiểm tra phản ứng giảm
                    if df.iloc[i+1]['close'] < current['close']:
                        zones.append({
                            'type': 'supply',
                            'price': current['high'],
                            'time': current['time']
                        })
                
                # Kiểm tra vùng demand
                if (current['low'] < df.iloc[i-2]['low'] and 
                    current['low'] < df.iloc[i-1]['low'] and 
                    current['low'] < df.iloc[i+1]['low'] and 
                    current['low'] < df.iloc[i+2]['low']):
                    
                    # Kiểm tra phản ứng tăng
                    if df.iloc[i+1]['close'] > current['close']:
                        zones.append({
                            'type': 'demand',
                            'price': current['low'],
                            'time': current['time']
                        })
            
            return zones, ""
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error in identify_zones: {error_msg}")
            return [], error_msg

    def check_breakout(self, price, zones, trend):
        """Kiểm tra xem có breakout không"""
        for zone in zones:
            if zone['type'] == 'supply' and trend == 'bearish':
                if price < zone['price'] * (1 - self.zone_threshold):
                    return True
            elif zone['type'] == 'demand' and trend == 'bullish':
                if price > zone['price'] * (1 + self.zone_threshold):
                    return True
        return False

    def check_candle_pattern(self, df):
        """Kiểm tra mô hình nến (pin bar hoặc engulfing)"""
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Kiểm tra pin bar
        if abs(last['open'] - last['close']) < self.price_threshold * last['close']:
            if last['high'] - max(last['open'], last['close']) > 2 * abs(last['open'] - last['close']):
                return 'pin_bar'
            if min(last['open'], last['close']) - last['low'] > 2 * abs(last['open'] - last['close']):
                return 'pin_bar'
        
        # Kiểm tra engulfing
        if (last['open'] < prev['close'] and 
            last['close'] > prev['open'] and 
    def run_strategy(self):
        """Chạy chiến lược giao dịch"""
        while True:
            try:
                # Lấy dữ liệu
                m15_df = self.utils.get_data(self.symbol, timeframes['M15'], 200)
                h4_df = self.utils.get_data(self.symbol, timeframes['H4'], 100)
                
                # Xác định xu hướng
                weekly_trend = self.utils.calculate_trend('D1')
                
                # Kiểm tra breakout
                zones = self.utils.identify_zones('H4')
                price = h4_df['close'].iloc[-1]
                
                if self.utils.check_breakout(price, zones, weekly_trend):
                    # Kiểm tra RSI và MACD
                    rsi_short, rsi_medium, rsi_long = self.indicators.calculate_rsi_indicators(m15_df)
                    macd, macd_signal = self.indicators.calculate_macd(m15_df)
                    prev_macd, prev_signal = self.indicators.calculate_macd(m15_df.iloc[:-1])
                    
                    # Kiểm tra tín hiệu RSI
                    rsi_signal, _ = self.indicators.check_rsi_signals(rsi_short, rsi_medium, rsi_long, weekly_trend)
                    
                    if rsi_signal:
                        if weekly_trend == 'bullish' and prev_macd < prev_signal and macd > macd_signal:
                            print("Bullish signal confirmed")
                            entry, sl, tp = self.backtest.calculate_levels(price, weekly_trend)
                            if self.trade_manager.execute_trade(weekly_trend, entry, sl, tp):
                                print(f"Buy order executed: Entry={entry}, SL={sl}, TP={tp}")
                        elif weekly_trend == 'bearish' and prev_macd > prev_signal and macd < macd_signal:
                            print("Bearish signal confirmed")
                            entry, sl, tp = self.backtest.calculate_levels(price, weekly_trend)
                            if self.trade_manager.execute_trade(weekly_trend, entry, sl, tp):
                                print(f"Sell order executed: Entry={entry}, SL={sl}, TP={tp}")
                

if __name__ == "__main__":
    strategy = AdvancedTradingStrategy(symbol="XAUUSD")
    strategy.run_strategy()
    supply_zones = []
    demand_zones = []

    
    for i in range(len(df_h4) - n_candles):
        high = df_h4['high'].iloc[i:i+n_candles].max()
        low = df_h4['low'].iloc[i:i+n_candles].min()
        if df_h4['close'].iloc[i+n_candles-1] < df_h4['open'].iloc[i+n_candles-1]:
            supply_zones.append((high, high * 1.005))  # Vùng supply ±0.5%
        else:
            demand_zones.append((low * 0.995, low))    # Vùng demand ±0.5%
    
    return supply_zones, demand_zones

def check_breakout(df_h4, supply_zones, demand_zones):
    last_price = df_h4['close'].iloc[-1]
    for zone in demand_zones:
        if zone[0] < last_price < zone[1]:
            return "bullish_breakout"
    for zone in supply_zones:
        if zone[0] < last_price < zone[1]:
            return "bearish_breakout"
    return None

def confirm_entry(df_m15, trend, breakout_type):
    # Kiểm tra pullback và xác nhận mô hình nến trên M15
    last_close = df_m15['close'].iloc[-1]
    last_open = df_m15['open'].iloc[-1]
    tp_points = 1000  # Take profit $10 (1:2 RR)
    trail_points = 300  # Trailing stop $3
    
    if not initialize_mt5():
        return
    
    try:
        while True:
            # 1. Phân tích xu hướng W1 và D1
            df_weekly = get_data(symbol, mt5.TIMEFRAME_W1, 200)
            df_daily = get_data(symbol, mt5.TIMEFRAME_D1, 200)
            trend = determine_trend(df_daily, df_weekly)
            print(f"Trend: {trend}")
            
            # 2. Xác định vùng supply/demand trên H4
            df_h4 = get_data(symbol, mt5.TIMEFRAME_H4, 100)
            supply_zones, demand_zones = identify_zones(df_h4)
            print(f"Supply zones: {len(supply_zones)}, Demand zones: {len(demand_zones)}")
            
            # 3. Kiểm tra breakout
            breakout_type = check_breakout(df_h4, supply_zones, demand_zones)
            
            # 4. Tinh chỉnh điểm vào lệnh trên M15
            if breakout_type:
                df_m15 = get_data(symbol, mt5.TIMEFRAME_M15, 50)
                entry_signal = confirm_entry(df_m15, trend, breakout_type)
                
                # 5. Đặt lệnh và quản lý rủi ro
                if entry_signal and not mt5.positions_get(symbol=symbol):
                    point = mt5.symbol_info(symbol).point
                    last_price = df_m15['close'].iloc[-1]
                    
                    if entry_signal == "buy":
                        sl = last_price - sl_points * point
                        tp = last_price + tp_points * point
                        place_order(symbol, mt5.ORDER_TYPE_BUY, volume, 
                                   mt5.symbol_info_tick(symbol).ask, sl, tp)
                    elif entry_signal == "sell":
                        sl = last_price + sl_points * point
                        tp = last_price - tp_points * point
                        place_order(symbol, mt5.ORDER_TYPE_SELL, volume, 
                                   mt5.symbol_info_tick(symbol).bid, sl, tp)
            
            # 6. Theo dõi và điều chỉnh
            positions = mt5.positions_get(symbol=symbol)
            for pos in positions:
                trail_stop(symbol, pos, trail_points)
            
            print(f"Time: {datetime.datetime.now()}, Price: {df_h4['close'].iloc[-1]:.2f}")
            time.sleep(900)  # Chờ 15 phút
            
    except KeyboardInterrupt:
        print("Dừng chiến lược")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()