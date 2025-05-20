"""
Trade management module.

This module provides functionality for managing trades, including order placement,
position tracking, and trade execution.
"""

import logging
import MetaTrader5 as mt5
from typing import Optional, Tuple, Dict, Any


class TradeManager:
    """
    Manages trade execution and position tracking.
    """
    
    def __init__(self, symbol: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the trade manager.
        
        Args:
            symbol: Trading symbol
            logger: Optional logger instance
        """
        self.symbol = symbol
        self.logger = logger or logging.getLogger(__name__)
        self.open_trades = []
        
    def place_order(
        self,
        order_type: int,
        volume: float,
        price: float,
        sl: float,
        tp: float,
        strategy_name: str,
        rsi_values: Tuple[Optional[float], Optional[float], Optional[float]] = (None, None, None),
        macd_crossover: bool = False,
        breakout_distance: float = 0.0
    ) -> bool:
        """
        Place a new order.
        
        Args:
            order_type: Order type (BUY/SELL)
            volume: Trade volume
            price: Entry price
            sl: Stop loss
            tp: Take profit
            strategy_name: Name of the strategy
            rsi_values: Tuple of RSI values (short, medium, long)
            macd_crossover: Whether MACD crossover occurred
            breakout_distance: Distance from breakout level
            
        Returns:
            bool: True if order placed successfully
        """
        try:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 234000,
                "comment": f"{strategy_name}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order failed: {result.comment}")
                return False
                
            self.logger.info(
                f"Order placed: {order_type}, Price={price}, SL={sl}, TP={tp}, "
                f"Strategy={strategy_name}, RSI={rsi_values}, MACD={macd_crossover}, "
                f"Breakout={breakout_distance}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return False
            
    def close_position(self, position_id: int) -> bool:
        """
        Close an open position.
        
        Args:
            position_id: Position ID to close
            
        Returns:
            bool: True if position closed successfully
        """
        try:
            position = mt5.positions_get(ticket=position_id)
            if not position:
                self.logger.error(f"Position {position_id} not found")
                return False
                
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": position[0].volume,
                "type": mt5.ORDER_TYPE_SELL if position[0].type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": position_id,
                "price": mt5.symbol_info_tick(self.symbol).bid if position[0].type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).ask,
                "deviation": 20,
                "magic": 234000,
                "comment": "Close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Close position failed: {result.comment}")
                return False
                
            self.logger.info(f"Position {position_id} closed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            return False
            
    def get_open_positions(self) -> list[Dict[str, Any]]:
        """
        Get all open positions.
        
        Returns:
            List of dictionaries containing position information
        """
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None:
                return []
                
            return [{
                'ticket': pos.ticket,
                'type': pos.type,
                'volume': pos.volume,
                'price_open': pos.price_open,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit,
                'comment': pos.comment
            } for pos in positions]
            
        except Exception as e:
            self.logger.error(f"Error getting open positions: {str(e)}")
            return []
