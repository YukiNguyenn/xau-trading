"""Quản lý giao dịch"""

import MetaTrader5 as mt5
import pandas as pd
from typing import Dict, List, Tuple

class TradeManager:
    def __init__(self, symbol: str, risk_params: Dict, trading_params: Dict):
        self.symbol = symbol
        self.risk_params = risk_params
        self.trading_params = trading_params
        self.open_tickets = set()
        
    def calculate_volume(self, account_balance: float, risk_percent: float = 1) -> float:
        """Tính toán kích thước lô dựa trên tài khoản và rủi ro"""
        risk_amount = account_balance * (risk_percent / 100)
        volume = risk_amount / (self.risk_params['stop_loss_points'] * 0.1)
        return min(volume, 1.0)  # Giới hạn tối đa 1 lot

    def execute_trade(self, trend: str, entry: float, sl: float, tp: float) -> bool:
        """Thực hiện giao dịch"""
        account_info = mt5.account_info()
        if not account_info:
            print("Failed to get account info")
            return False
            
        volume = self.calculate_volume(account_info.balance)
        price = mt5.symbol_info_tick(self.symbol).ask if trend == 'bullish' else mt5.symbol_info_tick(self.symbol).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if trend == 'bullish' else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": "Advanced Strategy",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed, retcode={result.retcode}")
            return False
            
        self.open_tickets.add(result.order)
        print(f"Order executed successfully: {result}")
        return True

    def update_trailing_stop(self, ticket: int) -> bool:
        """Cập nhật trailing stop cho lệnh đang mở"""
        position = mt5.positions_get(ticket=ticket)[0]
        if position:
            current_price = mt5.symbol_info_tick(self.symbol).ask if position.type == 0 else mt5.symbol_info_tick(self.symbol).bid
            
            sl = position.sl
            if position.type == 0:  # Buy
                if current_price > position.price_open + self.trading_params['trailing_stop_points'] * self.point:
                    sl = max(sl, current_price - self.risk_params['stop_loss_points'] * self.point)
            else:  # Sell
                if current_price < position.price_open - self.trading_params['trailing_stop_points'] * self.point:
                    sl = min(sl, current_price + self.risk_params['stop_loss_points'] * self.point)
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": sl,
                "tp": position.tp
            }
            
            result = mt5.order_send(request)
            return result.retcode == mt5.TRADE_RETCODE_DONE
        return False

    def check_open_positions(self) -> List[Dict]:
        """Kiểm tra và cập nhật các lệnh đang mở"""
        positions = mt5.positions_get()
        if not positions:
            return []
            
        history = mt5.history_deals_get(datetime.now() - timedelta(hours=1), datetime.now())
        if not history:
            return []
            
        closed_positions = []
        for deal in history:
            if deal.position_id in self.open_tickets:
                position = mt5.positions_get(ticket=deal.position_id)
                if not position:  # Lệnh đã đóng
                    closed_positions.append({
                        'ticket': deal.position_id,
                        'profit': deal.profit,
                        'close_time': deal.time
                    })
                    self.open_tickets.remove(deal.position_id)
        
        return closed_positions
