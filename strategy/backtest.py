"""Module backtest cho chiến lược giao dịch"""

import pandas as pd
import numpy as np
import ta
import json
from datetime import datetime
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

class Backtest:
    def __init__(self, symbol: str, risk_params: Dict, trading_params: Dict, position_params: Dict):
        self.symbol = symbol
        self.risk_params = risk_params
        self.trading_params = trading_params
        self.position_params = position_params
        self.point = None
        self.spread_points = trading_params.get('spread_points', 20)
        self.commission = trading_params.get('commission', 0.0001)
        self.trailing_stop_points = trading_params.get('trailing_stop_points', 300)
        self.stop_loss_points = risk_params.get('stop_loss_points', 500)
        self.take_profit_points = risk_params.get('take_profit_points', 1000)
        self.max_open_positions = position_params.get('max_open_positions', 3)
        self.priority_levels = position_params.get('priority_levels', {
            'high': {'rsi_short': 80, 'rsi_medium': 70, 'rsi_long': 60, 'macd_crossover': True},
            'medium': {'rsi_short': 75, 'rsi_medium': 65, 'rsi_long': 55, 'macd_crossover': True},
            'low': {'rsi_short': 70, 'rsi_medium': 60, 'rsi_long': 50, 'macd_crossover': False}
        })
        
    def get_point(self):
        """Lấy point từ MT5 và kiểm tra lỗi"""
        point = self.point
        if point is None:
            point = mt5.symbol_info(self.symbol).point
            if point is None:
                print(f"Failed to get point for symbol {self.symbol}")
                return None
            self.point = point
        return point
        
    def calculate_levels(self, price: float, trend: str) -> Tuple[float, float, float]:
        """Tính các mức vào lệnh với kiểm tra point"""
        point = self.get_point()
        if point is None:
            return None, None, None
            
        if trend == 'bullish':
            entry = price
            sl = price - self.stop_loss_points * point
            tp = price + self.take_profit_points * point
        else:
            entry = price
            sl = price + self.stop_loss_points * point
            tp = price - self.take_profit_points * point
        return entry, sl, tp

    def benchmark_backtest(self, df: pd.DataFrame, symbol: str, timeframe: str = 'H4') -> Dict:
        """
        Chạy backtest và đo thời gian thực thi
        Args:
            df: DataFrame chứa dữ liệu lịch sử
            symbol: Biểu đồ giao dịch
            timeframe: Khung thời gian (default: H4)
        Returns:
            Dict chứa kết quả backtest và thời gian thực thi
        """
        import time
        
        # Lấy point từ symbol
        self.point = df['point'].iloc[0] if 'point' in df.columns else 0.0001
        
        start_time = time.time()
        
        results = self.backtest(df, symbol, timeframe)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        results['execution_time'] = execution_time
        return results

    def backtest(self, df: pd.DataFrame, symbol: str, timeframe: str = 'H4', plot_results: bool = True) -> Dict:
        """
        Chạy backtest với giới hạn vị thế và ưu tiên
        Args:
            df: DataFrame chứa dữ liệu lịch sử
            symbol: Biểu đồ giao dịch
            timeframe: Khung thời gian (default: H4)
        Returns:
            Dict chứa kết quả backtest
        """
        results = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'net_profit': 0,
            'max_drawdown': 0,
            'trades': [],
            'open_positions': 0,
            'rejected_trades': 0,
            'avg_priority': 0,
            'high_priority_trades': 0,
            'medium_priority_trades': 0,
            'low_priority_trades': 0
        }
        
        # Kiểm tra và lấy point
        self.point = df['point'].iloc[0] if 'point' in df.columns else 0.0001
        if self.point is None:
            print("Failed to get point value")
            return results

        # Tính RSI và MACD
        df['rsi_short'] = df.ta.rsi(length=6)
        df['rsi_medium'] = df.ta.rsi(length=14)
        df['rsi_long'] = df.ta.rsi(length=24)
        df['macd'], df['macd_signal'], df['macd_hist'] = df.ta.macd(fast=12, slow=26, signal=9)

        # Tính các mức vào lệnh
        df['entry'], df['sl'], df['tp'] = zip(*df.apply(
            lambda row: self.calculate_levels(row['close'], 'bullish'),
            axis=1
        ))

        # Kiểm tra lỗi trong tính toán
        if df['entry'].isnull().any() or df['sl'].isnull().any() or df['tp'].isnull().any():
            print("Error in calculating entry levels")
            return results

        # Danh sách các vị thế đang mở
        open_trades = []
        total_priority = 0

        try:
            for i in range(len(df)):
                if i + 50 >= len(df):
                    break
                    
                current_row = df.iloc[i]
                future_prices = df.iloc[i+1:i+51]['close']
                
                # Kiểm tra lỗi trong dữ liệu tương lai
                if future_prices.isnull().any():
                    continue
                    
                # Tính khoảng cách breakout
                breakout_distance = abs(current_row['close'] - current_row['ema200'])
                
                # Xác định cấp độ ưu tiên
                priority = self._get_priority(
                    current_row['rsi_short'],
                    current_row['rsi_medium'],
                    current_row['rsi_long'],
                    current_row['macd'] > current_row['macd_signal'],
                    breakout_distance
                )
                
                # Cập nhật thống kê priority
                total_priority += self.priority_levels[priority]['weight']
                if priority == 'high':
                    results['high_priority_trades'] += 1
                elif priority == 'medium':
                    results['medium_priority_trades'] += 1
                else:
                    results['low_priority_trades'] += 1
                
                # Kiểm tra và đóng các vị thế có ưu tiên thấp hơn
                if len(open_trades) >= self.max_open_positions and priority != 'low':
                    self._close_low_priority_trades(open_trades, priority)
                    
                # Kiểm tra các điều kiện vào lệnh
                if (current_row['rsi_short'] < 30 and
                    current_row['rsi_medium'] < 30 and
                    current_row['rsi_long'] < 30 and
                    current_row['macd'] > current_row['macd_signal']):
                    
                    # Kiểm tra xem còn chỗ cho lệnh mới không
                    if len(open_trades) >= self.max_open_positions:
                        results['rejected_trades'] += 1
                        continue
                        
                    # Giả lập giao dịch
                    profit, success = self.simulate_trade(
                        current_row['entry'],
                        current_row['sl'],
                        current_row['tp'],
                        'bullish',
                        future_prices
                    )
                    
                    # Kiểm tra lỗi trong giao dịch
                    if not success:
                        continue
                    
                    results['total_trades'] += 1
                    results['gross_profit'] += profit if profit > 0 else 0
                    results['gross_loss'] += profit if profit < 0 else 0
                    results['net_profit'] += profit
                    
                    if profit > 0:
                        results['winning_trades'] += 1
                    else:
                        results['losing_trades'] += 1
                        
                    # Lưu thông tin giao dịch và thêm vào danh sách vị thế đang mở
                    trade = {
                        'time': current_row['time'],
                        'entry': current_row['entry'],
                        'sl': current_row['sl'],
                        'tp': current_row['tp'],
                        'profit': profit,
                        'priority': priority
                    }
                    results['trades'].append(trade)
                    open_trades.append(trade)

            # Tính max drawdown
            equity = [0]
            for trade in results['trades']:
                equity.append(equity[-1] + trade['profit'])
            results['max_drawdown'] = min(equity) - max(equity)
            
            # Tính trung bình priority
            if results['total_trades'] > 0:
                results['avg_priority'] = total_priority / results['total_trades']

        except Exception as e:
            print(f"Error during backtest: {str(e)}")
            return results

        return results

    def _get_priority(self, rsi_short: float, rsi_medium: float, rsi_long: float, macd_crossover: bool, breakout_distance: float) -> str:
        """Xác định cấp độ ưu tiên của lệnh dựa trên RSI và MACD"""
        for level in ['high', 'medium', 'low']:
            level_config = self.priority_levels[level]
            if (rsi_short <= level_config['rsi_short'] and
                rsi_medium <= level_config['rsi_medium'] and
                rsi_long <= level_config['rsi_long'] and
                (level_config['macd_crossover'] or macd_crossover)):
                return level
        return 'low'

    def _close_low_priority_trades(self, open_trades: List[Dict], new_priority: str):
        """Đóng các vị thế có ưu tiên thấp hơn vị thế mới"""
        # Sắp xếp các vị thế theo ưu tiên từ thấp đến cao
        open_trades.sort(key=lambda x: self.priority_levels[x['priority']]['weight'])
        
        # Đóng các vị thế có ưu tiên thấp hơn
        for trade in open_trades[:]:
            if self.priority_levels[trade['priority']]['weight'] < self.priority_levels[new_priority]['weight']:
                self.close_position(trade['ticket'])
                open_trades.remove(trade)
                
    def plot_backtest_results(self, results: Dict):
        """
        Vẽ biểu đồ equity curve và phân phối lợi nhuận
        Args:
            results: Kết quả backtest
        """
        if not results['trades']:
            print("Không có giao dịch để vẽ biểu đồ")
            return

        # Tính equity curve
        equity = [0]
        times = []
        profits = []
        
        for trade in results['trades']:
            equity.append(equity[-1] + trade['profit'])
            times.append(trade['time'])
            profits.append(trade['profit'])

        # Tạo figure với kích thước lớn
        plt.figure(figsize=(14, 10))
        
        # Biểu đồ equity curve
        plt.subplot(2, 1, 1)
        plt.plot(times, equity[1:], label='Equity Curve')
        
        # Định dạng ngày tháng
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()
        
        # Định dạng số tiền
        def money_formatter(x, pos):
            return f'${x:,.0f}'
        plt.gca().yaxis.set_major_formatter(FuncFormatter(money_formatter))
        
        # Thêm các chỉ số thống kê vào title
        plt.title(f'Backtest Results - {self.symbol}\n'
                  f'Win Rate: {results["win_rate"]:.1%} | '
                  f'Profit Factor: {results["profit_factor"]:.2f} | '
                  f'Avg Trade: ${results["avg_trade_profit"]:.2f} | '
                  f'Total Trades: {results["total_trades"]}\n'
                  f'High Priority: {results["high_priority_trades"]} | '
                  f'Medium Priority: {results["medium_priority_trades"]} | '
                  f'Low Priority: {results["low_priority_trades"]}')
        
        plt.xlabel('Time')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(True)
        
        # Biểu đồ phân phối lợi nhuận
        plt.subplot(2, 1, 2)
        plt.hist(profits, bins=30, edgecolor='black', alpha=0.7)
        plt.title('Profit Distribution')
        plt.xlabel('Profit ($)')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Thêm đường thẳng thể hiện lợi nhuận trung bình
        avg_profit = sum(profits) / len(profits)
        plt.axvline(avg_profit, color='red', linestyle='dashed', linewidth=2)
        plt.text(avg_profit, plt.ylim()[1]*0.9, f'Avg Profit: ${avg_profit:.2f}', 
                 rotation=90, verticalalignment='top', color='red')
        
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=300)
        print("Đã lưu biểu đồ vào file backtest_results.png")
        plt.close()

    def simulate_trade(self, entry: float, sl: float, tp: float, trend: str, 
                      future_prices: pd.Series) -> Tuple[float, bool]:
        """Giả lập kết quả giao dịch với giới hạn và kiểm tra lỗi"""
        point = self.get_point()
        if point is None:
            return 0, False
            
        # Giới hạn future_prices tối đa 50 nến
        future_prices = future_prices[:50]
        if len(future_prices) == 0:
            return 0, False
            
        profit = 0
        trade_closed = False
        current_sl = sl
        
        for j, future_price in enumerate(future_prices):
            # Kiểm tra trailing stop
            if trend == 'bullish':
                if future_price > entry + self.trailing_stop_points * point:
                    current_sl = max(current_sl, future_price - self.stop_loss_points * point)
            else:
                if future_price < entry - self.trailing_stop_points * point:
                    current_sl = min(current_sl, future_price + self.stop_loss_points * point)
            
            # Kiểm tra các điều kiện đóng lệnh
            if trend == 'bullish':
                if future_price >= tp:
                    profit = (tp - entry - self.spread_points * point) * 0.1
                    trade_closed = True
                    break
                elif future_price <= current_sl:
                    profit = (current_sl - entry - self.spread_points * point) * 0.1
                    trade_closed = True
                    break
            else:
                if future_price <= tp:
                    profit = (entry - tp - self.spread_points * point) * 0.1
                    trade_closed = True
                    break
                elif future_price >= current_sl:
                    profit = (entry - current_sl - self.spread_points * point) * 0.1
                    trade_closed = True
                    break
            
            # Kiểm tra thời gian tối đa (1 tuần, 24 nến H4)
            if j >= 24:
                if trend == 'bullish':
                    profit = (future_price - entry - self.spread_points * point) * 0.1
                else:
                    profit = (entry - future_price - self.spread_points * point) * 0.1
                trade_closed = True
                break
        
        # Nếu không đóng trong 50 nến, tính lợi nhuận tại nến cuối
        if not trade_closed:
            if trend == 'bullish':
                profit = (future_prices.iloc[-1] - entry - self.spread_points * point) * 0.1
            else:
                profit = (entry - future_prices.iloc[-1] - self.spread_points * point) * 0.1
            trade_closed = True
        
        # Tính toán phí giao dịch
        commission = entry * self.commission
        profit -= commission
        
        return profit, trade_closed
        """Giả lập kết quả giao dịch"""
        profit = 0
        trade_closed = False
        current_sl = sl
        
        # Giới hạn future_prices tối đa 50 nến
        future_prices = future_prices[:50]
        
        for j, future_price in enumerate(future_prices):
            # Kiểm tra trailing stop
            if trend == 'bullish':
                if future_price > entry + self.trailing_stop_points * self.point:
                    current_sl = max(current_sl, future_price - self.stop_loss_points * self.point)
            else:
                if future_price < entry - self.trailing_stop_points * self.point:
                    current_sl = min(current_sl, future_price + self.stop_loss_points * self.point)
            
            # Kiểm tra các điều kiện đóng lệnh
            if trend == 'bullish':
                if future_price >= tp:
                    profit = (tp - entry - self.spread_points * self.point) * 0.1
                    trade_closed = True
                    break
                elif future_price <= current_sl:
                    profit = (current_sl - entry - self.spread_points * self.point) * 0.1
                    trade_closed = True
                    break
            else:
                if future_price <= tp:
                    profit = (entry - tp - self.spread_points * self.point) * 0.1
                    trade_closed = True
                    break
                elif future_price >= current_sl:
                    profit = (entry - current_sl - self.spread_points * self.point) * 0.1
                    trade_closed = True
                    break
            
            # Kiểm tra thời gian tối đa (1 tuần, 24 nến H4)
            if j >= 24:
                if trend == 'bullish':
                    profit = (future_price - entry - self.spread_points * self.point) * 0.1
                else:
                    profit = (entry - future_price - self.spread_points * self.point) * 0.1
                trade_closed = True
                break
        
        # Nếu không đóng trong 50 nến, tính lợi nhuận tại nến cuối
        if not trade_closed and len(future_prices) > 0:
            if trend == 'bullish':
                profit = (future_prices.iloc[-1] - entry - self.spread_points * self.point) * 0.1
            else:
                profit = (entry - future_prices.iloc[-1] - self.spread_points * self.point) * 0.1
            trade_closed = True
        
        # Tính toán phí giao dịch
        commission = entry * self.commission
        profit -= commission
        
        return profit, trade_closed
            if trend == 'bullish':
                if future_price > entry + self.trading_params['trailing_stop_points'] * self.point:
                    current_sl = max(current_sl, future_price - self.risk_params['stop_loss_points'] * self.point)
            else:
                if future_price < entry - self.trading_params['trailing_stop_points'] * self.point:
                    current_sl = min(current_sl, future_price + self.risk_params['stop_loss_points'] * self.point)
            
            # Kiểm tra các điều kiện đóng lệnh
            if (trend == 'bullish' and future_price >= tp) or \
               (trend == 'bearish' and future_price <= tp):
                profit = self.calculate_profit(entry, tp, trend)
                trade_closed = True
                break
            elif (trend == 'bullish' and future_price <= current_sl) or \
                 (trend == 'bearish' and future_price >= current_sl):
                profit = self.calculate_profit(entry, current_sl, trend)
                trade_closed = True
                break
            
            # Kiểm tra thời gian tối đa
            if j >= self.trading_params['max_week_duration']:
                profit = self.calculate_profit(entry, future_price, trend)
                trade_closed = True
                break
        
        # Nếu không đóng trong thời gian tối đa
        if not trade_closed and len(future_prices) > 0:
            profit = self.calculate_profit(entry, future_prices.iloc[-1], trend)
            trade_closed = True
        
        return profit, trade_closed

    def calculate_profit(self, entry: float, exit: float, trend: str) -> float:
        """Tính lợi nhuận của lệnh giao dịch"""
        if trend == 'bullish':
            profit = (exit - entry - self.trading_params['spread_points'] * self.point) * 0.1
        else:
            profit = (entry - exit - self.trading_params['spread_points'] * self.point) * 0.1
        
        # Trừ commission
        profit -= entry * self.trading_params['commission']
        return profit

    def calculate_stats(self, trades: List[Dict]) -> Dict:
        """Tính các chỉ số thống kê"""
        total_trades = len(trades)
        if total_trades == 0:
            return {
                'win_rate': 0,
                'profit_factor': float('inf'),
                'avg_duration': 0
            }
            
        winning_trades = len([t for t in trades if t['profit'] > 0])
        losing_trades = len([t for t in trades if t['profit'] < 0])
        
        win_rate = winning_trades / total_trades
        profit_factor = abs(sum(t['profit'] for t in trades if t['profit'] > 0) / 
                          sum(t['profit'] for t in trades if t['profit'] < 0)) if losing_trades > 0 else float('inf')
        
        avg_duration = sum(t['duration'] for t in trades) / total_trades
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_duration': avg_duration
        }
