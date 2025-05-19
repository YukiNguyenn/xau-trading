import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from pandas_ta import rsi, macd
import json
import os

class AdvancedTradingStrategy:
    def __init__(self, symbol=None):
        # Load configuration from file
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Trading settings
        self.symbol = symbol if symbol else self.config['trading']['symbol']
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
        self.spread_points = self.config['costs']['spread']['max_points']  # Updated line
        self.commission = self.config['costs']['commission']
        
        # Position management settings
        position_config = self.config['position_management']
        self.max_open_positions = position_config['max_open_positions']
        self.priority_levels = position_config['priority_levels']
        
        # MT5 account settings
        self.account = self.config['mt5_account']['account']
        self.password = self.config['mt5_account']['password']
        self.server = self.config['mt5_account']['server']
        
        # Initialize MT5 connection
        if not self.initialize_mt5():
            quit()
        print("MT5 initialized and logged in successfully")

        # Create trade log file if it doesn't exist
        try:
            with open('trade_log.csv', 'r') as f:
                pass
        except FileNotFoundError:
            with open('trade_log.csv', 'w') as f:
                f.write("time,ticket,entry,sl,tp,result\n")

        # Track open tickets and trades
        self.open_tickets = set()
        self.open_trades = []
        
        # Backtesting attributes
        self.initial_balance = 10000  # Starting balance for backtesting
        self.commission_per_trade = 0.0001  # 0.01% commission per trade
        self.slippage = 0.0001  # 0.01% slippage per trade

    def initialize_mt5(self):
        """Initialize and login to MT5 platform."""
        try:
            if not mt5.initialize():
                print(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            if not mt5.login(self.account, password=self.password, server=self.server):
                print(f"MT5 login failed: {mt5.last_error()}")
                return False
            return True
        except Exception as e:
            print(f"Error initializing MT5: {str(e)}")
            return False

    def _get_priority(self, rsi_short, rsi_medium, rsi_long, macd_crossover, breakout_distance):
        """Determine trade priority based on indicators and breakout distance."""
        min_score = self.priority_levels['high']['breakout_min_score']
        breakout_score = max(min(breakout_distance / 0.01, 1), min_score)
        
        for level in ['high', 'medium', 'low']:
            level_config = self.priority_levels[level]
            if (rsi_short <= level_config['rsi_short'] and
                rsi_medium <= level_config['rsi_medium'] and
                rsi_long <= level_config['rsi_long'] and
                (level_config['macd_crossover'] or macd_crossover) and
                breakout_score >= level_config['breakout_min_score']):
                return level
        return 'low'

    def _close_low_priority_trades(self, new_priority):
        """Close trades with lower priority to make room for new trade."""
        for trade in self.open_trades[:]:
            if trade['priority'] < new_priority:
                self.close_position(trade['ticket'])
                self.open_trades.remove(trade)

    def close_position(self, ticket):
        """Close a specific position with error handling."""
        try:
            if not mt5.initialize():
                print(f"MT5 initialization failed: {mt5.last_error()}")
                return False
                
            if not mt5.login(self.account, password=self.password, server=self.server):
                print(f"MT5 login failed: {mt5.last_error()}")
                return False
                
            position = mt5.positions_get(ticket=ticket)
            if not position:
                print(f"Position {ticket} not found")
                return False
                
            position = position[0]
            symbol_info = mt5.symbol_info_tick(self.symbol)
            if not symbol_info:
                print(f"Failed to get symbol info for {self.symbol}")
                return False
                
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": symbol_info.ask if position.type == mt5.ORDER_TYPE_BUY else symbol_info.bid,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
                "position": ticket
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Position {ticket} closed successfully")
                self.open_tickets.remove(ticket)
                return True
                
            print(f"Failed to close position {ticket}: {mt5.last_error()}")
            return False
            
        except Exception as e:
            print(f"Error closing position {ticket}: {str(e)}")
            return False

    def place_order(self, order_type, volume, price, sl, tp, rsi_values, macd_crossover, breakout_distance):
        """Place a trade order with error handling and priority system."""
        try:
            priority = self._get_priority(*rsi_values, macd_crossover, breakout_distance)
            
            if len(self.open_trades) >= self.max_open_positions and priority != 'low':
                self._close_low_priority_trades(priority)
                
            if len(self.open_trades) >= self.max_open_positions:
                print("Maximum open positions reached")
                return False

            if not mt5.initialize():
                print(f"MT5 initialization failed: {mt5.last_error()}")
                return False
                
            if not mt5.login(self.account, password=self.password, server=self.server):
                print(f"MT5 login failed: {mt5.last_error()}")
                return False
                
            if volume < self.min_volume or volume > self.max_volume:
                print(f"Invalid volume: {volume}")
                return False
                
            if price <= 0 or sl <= 0 or tp <= 0:
                print(f"Invalid price: {price}, sl: {sl}, tp: {tp}")
                return False
                
            symbol_info = mt5.symbol_info(self.symbol)
            if not symbol_info:
                print(f"Failed to get symbol info for {self.symbol}")
                return False
                
            spread = symbol_info.spread
            if spread > self.spread_points:
                print(f"Current spread {spread} exceeds maximum allowed {self.spread_points}")
                return False
                
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
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
                print(f"Order failed: {mt5.last_error()}")
                return False
                
            if result.order in self.open_tickets:
                print(f"Duplicate ticket {result.order} detected")
                return False
                
            trade = {
                'ticket': result.order,
                'entry': price,
                'sl': sl,
                'tp': tp,
                'priority': priority,
                'profit': 0
            }
            self.open_tickets.add(result.order)
            self.open_trades.append(trade)
            
            print(f"Order placed successfully: {result.order}, Priority: {priority}")
            return True
            
        except Exception as e:
            print(f"Error placing order: {str(e)}")
            return False

    def get_data(self, timeframe, bars):
        """Fetch historical data from MT5."""
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframes[timeframe], 0, bars)
            df = pd.DataFrame(rates)
            if len(df) == 0:
                raise ValueError(f"No data received for {self.symbol} {timeframe}")
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
        except Exception as e:
            print(f"Error in get_data: {str(e)}")
            raise ValueError(f"Failed to get data: {str(e)}")

    def calculate_trend(self, timeframe):
        """Calculate trend based on EMA50 and EMA200."""
        try:
            df = self.get_data(timeframe, 200)
            if df is None or len(df) < 200:
                raise ValueError(f"Not enough data for {timeframe} timeframe")
                
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
        """Identify supply/demand zones on H4 timeframe."""
        try:
            df = self.get_data(timeframe, 100)
            if df is None or len(df) < 100:
                raise ValueError(f"Not enough data for {timeframe} timeframe")
                
            zones = []
            for i in range(2, len(df) - 2):
                current = df.iloc[i]
                
                if (current['high'] > df.iloc[i-2]['high'] and 
                    current['high'] > df.iloc[i-1]['high'] and 
                    current['high'] > df.iloc[i+1]['high'] and 
                    current['high'] > df.iloc[i+2]['high']):
                    if df.iloc[i+1]['close'] < current['close']:
                        zones.append({
                            'type': 'supply',
                            'price': current['high'],
                            'time': current['time']
                        })
                
                if (current['low'] < df.iloc[i-2]['low'] and 
                    current['low'] < df.iloc[i-1]['low'] and 
                    current['low'] < df.iloc[i+1]['low'] and 
                    current['low'] < df.iloc[i+2]['low']):
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
        """Check for breakout from supply/demand zones."""
        for zone in zones:
            if zone['type'] == 'supply' and trend == 'bearish':
                if price < zone['price'] * (1 - self.zone_threshold):
                    return True
            elif zone['type'] == 'demand' and trend == 'bullish':
                if price > zone['price'] * (1 + self.zone_threshold):
                    return True
        return False

    def check_candle_pattern(self, df):
        """Check for pin bar or engulfing candle patterns."""
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        if abs(last['open'] - last['close']) < self.price_threshold * last['close']:
            if last['high'] - max(last['open'], last['close']) > 2 * abs(last['open'] - last['close']):
                return 'pin_bar'
            if min(last['open'], last['close']) - last['low'] > 2 * abs(last['open'] - last['close']):
                return 'pin_bar'
        
        if (last['open'] < prev['close'] and 
            last['close'] > prev['open'] and 
            last['high'] > prev['high'] and 
            last['low'] < prev['low']):
            return 'engulfing'
        
        return None

    def calculate_rsi_indicators(self, df):
        """Calculate RSI for short, medium, and long periods."""
        rsi_short = rsi(df['close'], length=self.rsi_short).iloc[-1]
        rsi_medium = rsi(df['close'], length=self.rsi_medium).iloc[-1]
        rsi_long = rsi(df['close'], length=self.rsi_long).iloc[-1]
        return rsi_short, rsi_medium, rsi_long

    def check_rsi_signals(self, rsi_short, rsi_medium, rsi_long, trend):
        """Check RSI signals based on trend."""
        if trend == 'bullish':
            if (rsi_short < self.rsi_short_oversold and
                rsi_medium < self.rsi_medium_oversold and
                rsi_long < self.rsi_long_oversold):
                return True, 'buy'
        elif trend == 'bearish':
            if (rsi_short > self.rsi_short_overbought and
                rsi_medium > self.rsi_medium_overbought and
                rsi_long > self.rsi_long_overbought):
                return True, 'sell'
        return False, None

    def calculate_macd(self, df):
        """Calculate MACD and signal line."""
        macd_df = macd(df['close'], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        return macd_df['MACD_12_26_9'].iloc[-1], macd_df['MACDs_12_26_9'].iloc[-1]

    def calculate_levels(self, price, trend):
        """Calculate entry, stop loss, and take profit levels."""
        point = mt5.symbol_info(self.symbol).point
        if trend == 'bullish':
            entry = price
            sl = price - self.stop_loss_points * point
            tp = price + self.take_profit_points * point
        else:  # bearish
            entry = price
            sl = price + self.stop_loss_points * point
            tp = price - self.take_profit_points * point
        return entry, sl, tp

    def run_strategy(self):
        """Run the trading strategy."""
        while True:
            try:
                # Fetch data
                m15_df = self.get_data('M15', 200)
                h4_df = self.get_data('H4', 100)
                
                # Determine trend
                weekly_trend, _ = self.calculate_trend('D1')
                
                # Check breakout
                zones, _ = self.identify_zones('H4')
                price = h4_df['close'].iloc[-1]
                
                if self.check_breakout(price, zones, weekly_trend):
                    # Calculate indicators
                    rsi_short, rsi_medium, rsi_long = self.calculate_rsi_indicators(m15_df)
                    macd, macd_signal = self.calculate_macd(m15_df)
                    prev_macd, prev_signal = self.calculate_macd(m15_df.iloc[:-1])
                    
                    # Check RSI signals
                    rsi_signal, signal_type = self.check_rsi_signals(rsi_short, rsi_medium, rsi_long, weekly_trend)
                    
                    if rsi_signal:
                        if weekly_trend == 'bullish' and prev_macd < prev_signal and macd > macd_signal:
                            print("Bullish signal confirmed")
                            entry, sl, tp = self.calculate_levels(price, weekly_trend)
                            breakout_distance = abs(price - zones[0]['price']) if zones else 0
                            self.place_order(mt5.ORDER_TYPE_BUY, self.min_volume, entry, sl, tp,
                                           (rsi_short, rsi_medium, rsi_long), True, breakout_distance)
                            print(f"Buy order executed: Entry={entry}, SL={sl}, TP={tp}")
                        elif weekly_trend == 'bearish' and prev_macd > prev_signal and macd < macd_signal:
                            print("Bearish signal confirmed")
                            entry, sl, tp = self.calculate_levels(price, weekly_trend)
                            breakout_distance = abs(price - zones[0]['price']) if zones else 0
                            self.place_order(mt5.ORDER_TYPE_SELL, self.min_volume, entry, sl, tp,
                                           (rsi_short, rsi_medium, rsi_long), True, breakout_distance)
                            print(f"Sell order executed: Entry={entry}, SL={sl}, TP={tp}")
                
                # Sleep for 15 minutes
                time.sleep(900)
                
            except KeyboardInterrupt:
                print("Strategy stopped by user")
                break
            except Exception as e:
                print(f"Error in run_strategy: {str(e)}")
                time.sleep(60)  # Wait before retrying

def main():
    """Initialize and run the trading strategy."""
    strategy = AdvancedTradingStrategy(symbol="XAUUSD")
    strategy.run_strategy()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        mt5.shutdown()