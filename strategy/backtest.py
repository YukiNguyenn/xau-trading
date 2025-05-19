"""Module backtest cho chiến lược giao dịch"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

class Backtest:
    def __init__(self, symbol: str, risk_params: Dict, trading_params: Dict):
        self.symbol = symbol
        self.risk_params = risk_params
        self.trading_params = trading_params
        self.point = None

    def calculate_levels(self, price: float, trend: str) -> Tuple[float, float, float]:
        """Tính các mức vào lệnh"""
        if trend == 'bullish':
            entry = price
            sl = price - self.risk_params['stop_loss_points'] * self.point
            tp = price + self.risk_params['take_profit_points'] * self.point
        else:
            entry = price
            sl = price + self.risk_params['stop_loss_points'] * self.point
            tp = price - self.risk_params['take_profit_points'] * self.point
        return entry, sl, tp

    def simulate_trade(self, entry: float, sl: float, tp: float, trend: str, 
                      future_prices: pd.Series) -> Tuple[float, bool]:
        """Giả lập kết quả giao dịch"""
        profit = 0
        trade_closed = False
        current_sl = sl
        
        for j, future_price in enumerate(future_prices):
            # Kiểm tra trailing stop
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
