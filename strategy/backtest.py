"""Module backtest cho chiến lược giao dịch"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

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
        self.spread_points = trading_params.get('spread_points', 20)
        self.commission = trading_params.get('commission', 0.0001)
        self.trailing_stop_points = trading_params.get('trailing_stop_points', 300)
        self.stop_loss_points = risk_params.get('stop_loss_points', 500)
        self.take_profit_points = risk_params.get('take_profit_points', 1000)

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

    def backtest(self, df: pd.DataFrame, symbol: str, timeframe: str = 'H4') -> Dict:
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
            'rejected_trades': 0
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

        try:
            for i in range(len(df)):
                if i + 50 >= len(df):
                    break
                    
                current_row = df.iloc[i]
                future_prices = df.iloc[i+1:i+51]['close']
                
                # Kiểm tra lỗi trong dữ liệu tương lai
                if future_prices.isnull().any():
                    continue
                    
                # Xác định cấp độ ưu tiên
                priority = self._get_priority(current_row)
                
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

        except Exception as e:
            print(f"Error during backtest: {str(e)}")
            return results

        return results

    def _get_priority(self, row):
        """Xác định cấp độ ưu tiên của lệnh dựa trên RSI và MACD"""
        for level in ['high', 'medium', 'low']:
            level_config = self.priority_levels[level]
            if (row['rsi_short'] <= level_config['rsi_short'] and
                row['rsi_medium'] <= level_config['rsi_medium'] and
                row['rsi_long'] <= level_config['rsi_long'] and
                (level_config['macd_crossover'] or row['macd'] > row['macd_signal'])):
                return level
        return 'low'

    def _close_low_priority_trades(self, open_trades, new_priority):
        """Đóng các vị thế có ưu tiên thấp hơn để nhường chỗ cho lệnh mới"""
        for trade in open_trades[:]:  # Copy list để tránh lỗi khi thay đổi trong vòng lặp
            if trade['priority'] < new_priority and trade['profit'] > 0:
                open_trades.remove(trade)

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
