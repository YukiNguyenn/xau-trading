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
        self.stop_loss_points = config['trading']['stop_loss_points']
        self.take_profit_points = config['trading']['take_profit_points']
        self.trailing_stop_points = config['trading']['trailing_stop_points']
        self.min_volume = config['trading']['min_volume']
        self.max_volume = config['trading']['max_volume']
        self.price_threshold = config['trading']['price_threshold']
        self.zone_threshold = config['trading']['zone_threshold']
        
        # RSI settings
        rsi_config = config['indicators']['rsi']
        self.rsi_short_period = rsi_config['periods']['short']
        self.rsi_medium_period = rsi_config['periods']['medium']
        self.rsi_long_period = rsi_config['periods']['long']
        
        self.rsi_short_overbought = rsi_config['thresholds']['short']['overbought']
        self.rsi_short_oversold = rsi_config['thresholds']['short']['oversold']
        self.rsi_medium_overbought = rsi_config['thresholds']['medium']['overbought']
        self.rsi_medium_oversold = rsi_config['thresholds']['medium']['oversold']
        self.rsi_long_overbought = rsi_config['thresholds']['long']['overbought']
        self.rsi_long_oversold = rsi_config['thresholds']['long']['oversold']
        
        # MACD settings
        macd_config = config['indicators']['macd']
        self.macd_fast = macd_config['fast']
        self.macd_slow = macd_config['slow']
        self.macd_signal = macd_config['signal']
        
        # Cost settings
        self.spread_points = config['costs']['spread_points']
        self.commission = config['costs']['commission']
        
        # Khởi tạo kết nối MT5
        if not mt5.initialize():
            print("Initialize() failed, error code =", mt5.last_error())
            quit()
        print("MT5 initialized successfully")

        # Tạo file log nếu chưa tồn tại
        try:
            with open('trade_log.csv', 'r') as f:
                pass
        except FileNotFoundError:
            with open('trade_log.csv', 'w') as f:
                f.write("time,ticket,entry,sl,tp,result\n")

        # Danh sách ticket đang mở
        self.open_tickets = set()

        def get_data(self, timeframe, bars):
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframes[timeframe], 0, bars)
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df

        def calculate_trend(self, timeframe):
            """Tính toán xu hướng dựa trên EMA50 và EMA200"""
            df = self.get_data(timeframe, 200)
            
            # Tính EMA
            df['ema50'] = df['close'].ewm(span=50).mean()
            df['ema200'] = df['close'].ewm(span=200).mean()
            
            last_row = df.iloc[-1]
            
            if last_row['ema50'] > last_row['ema200']:
                return 'bullish'
            elif last_row['ema50'] < last_row['ema200']:
                return 'bearish'
            else:
                return 'neutral'

        def identify_zones(self, timeframe):
            """Xác định vùng supply/demand trên khung H4"""
            df = self.get_data(timeframe, 100)
            
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
            
            return zones

    def get_data(self, timeframe, bars):
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframes[timeframe], 0, bars)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def calculate_trend(self, timeframe):
        """Tính toán xu hướng dựa trên EMA50 và EMA200"""
        df = self.get_data(timeframe, 200)
        
        # Tính EMA
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['ema200'] = df['close'].ewm(span=200).mean()
        
        last_row = df.iloc[-1]
        
        if last_row['ema50'] > last_row['ema200']:
            return 'bullish'
        elif last_row['ema50'] < last_row['ema200']:
            return 'bearish'
        else:
            return 'neutral'

    def identify_zones(self, timeframe):
        """Xác định vùng supply/demand trên khung H4"""
        df = self.get_data(timeframe, 100)
        
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
        
        return zones

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
                
                # Kiểm tra và cập nhật các lệnh đang mở
                closed_positions = self.trade_manager.check_open_positions()
                for pos in closed_positions:
                    print(f"Position {pos['ticket']} closed with profit: {pos['profit']}")
                
                # Cập nhật trailing stop cho các lệnh đang mở
                # Tính toán các mức vào lệnh
                entry, sl, tp = self.calculate_levels(current_price, weekly_trend)
                
                # Thực hiện giao dịch
                self.execute_trade(weekly_trend, entry, sl, tp)
                
                # Cập nhật trailing stop
                positions = mt5.positions_get()
                if positions:
                    for position in positions:
                        self.update_trailing_stop(position.ticket)
                
                # Kiểm tra và ghi log các lệnh đã đóng
                positions = mt5.positions_get()
                if positions:
                    history = mt5.history_deals_get(datetime.now() - timedelta(hours=1), datetime.now())
                    if history:
                        for deal in history:
                            if deal.position_id in self.open_tickets:
                                position = mt5.positions_get(ticket=deal.position_id)
                                if not position:  # Lệnh đã đóng
                                    self.log_trade(
                                        deal.position_id,
                                        deal.price,
                                        deal.sl,
                                        deal.tp,
                                        f"CLOSED: {deal.profit}"
                                    )
                                    self.open_tickets.remove(deal.position_id)
                
                # Chờ 15 phút trước khi kiểm tra lại
                time.sleep(900)
                
            except Exception as e:
                print(f"Error: {str(e)}")
                time.sleep(900)

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
    
    if trend == "bullish" and breakout_type == "bullish_breakout":
        # Bullish pin bar hoặc engulfing
        if last_close > last_open and (last_close - last_open) / last_open > 0.001:
            return "buy"
    elif trend == "bearish" and breakout_type == "bearish_breakout":
        # Bearish pin bar hoặc engulfing
        if last_close < last_open and (last_open - last_close) / last_open > 0.001:
            return "sell"
    return None

def place_order(symbol, order_type, volume, price, sl, tp):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("Lệnh thất bại, retcode =", result.retcode)
        return False
    print(f"Lệnh {order_type} thành công, ticket =", result.order)
    return True

def trail_stop(symbol, position, trail_points):
    point = mt5.symbol_info(symbol).point
    if position.type == mt5.ORDER_TYPE_BUY:
        new_sl = mt5.symbol_info_tick(symbol).bid - trail_points * point
        if new_sl > position.sl:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position.ticket,
                "sl": new_sl,
                "tp": position.tp
            }
            mt5.order_send(request)
    elif position.type == mt5.ORDER_TYPE_SELL:
        new_sl = mt5.symbol_info_tick(symbol).ask + trail_points * point
        if new_sl < position.sl:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position.ticket,
                "sl": new_sl,
                "tp": position.tp
            }
            mt5.order_send(request)

def main():
    # Thông số chiến lược
    symbol = "XAUUSD"
    volume = 0.01  # Khối lượng nhỏ cho XAUUSD
    sl_points = 500  # Stop loss $5
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