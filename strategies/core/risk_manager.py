"""
Risk management module.

This module provides functionality for managing trading risk, including
position sizing, stop loss calculation, and risk assessment.
"""

import logging
from typing import Dict, Optional, Tuple


class RiskManager:
    """
    Manages trading risk and position sizing.
    """
    
    def __init__(
        self,
        account_balance: float,
        risk_percent: float = 1.0,
        max_positions: int = 3,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the risk manager.
        
        Args:
            account_balance: Current account balance
            risk_percent: Maximum risk per trade as percentage
            max_positions: Maximum number of concurrent positions
            logger: Optional logger instance
        """
        self.account_balance = account_balance
        self.risk_percent = risk_percent
        self.max_positions = max_positions
        self.logger = logger or logging.getLogger(__name__)
        
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        point_value: float = 0.1
    ) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            point_value: Value of one point
            
        Returns:
            float: Position size in lots
        """
        try:
            risk_amount = self.account_balance * (self.risk_percent / 100)
            stop_loss_points = abs(entry_price - stop_loss) / point_value
            position_size = risk_amount / (stop_loss_points * point_value)
            
            # Round to 2 decimal places and limit to 1 lot
            position_size = min(round(position_size, 2), 1.0)
            
            self.logger.info(
                f"Calculated position size: {position_size} lots "
                f"(Risk: {self.risk_percent}%, Stop Loss: {stop_loss_points} points)"
            )
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.01  # Return minimum position size on error
            
    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        atr_multiplier: float = 2.0
    ) -> float:
        """
        Calculate stop loss level based on ATR.
        
        Args:
            entry_price: Entry price
            direction: Trade direction ('buy' or 'sell')
            atr: Average True Range value
            atr_multiplier: ATR multiplier for stop loss
            
        Returns:
            float: Stop loss price
        """
        try:
            stop_distance = atr * atr_multiplier
            if direction.lower() == 'buy':
                stop_loss = entry_price - stop_distance
            else:
                stop_loss = entry_price + stop_distance
                
            self.logger.info(
                f"Calculated stop loss: {stop_loss} "
                f"(ATR: {atr}, Multiplier: {atr_multiplier})"
            )
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {str(e)}")
            return entry_price
            
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        risk_reward_ratio: float = 2.0
    ) -> float:
        """
        Calculate take profit level based on risk:reward ratio.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_reward_ratio: Desired risk:reward ratio
            
        Returns:
            float: Take profit price
        """
        try:
            risk = abs(entry_price - stop_loss)
            reward = risk * risk_reward_ratio
            
            if entry_price > stop_loss:  # Buy trade
                take_profit = entry_price + reward
            else:  # Sell trade
                take_profit = entry_price - reward
                
            self.logger.info(
                f"Calculated take profit: {take_profit} "
                f"(Risk:Reward = 1:{risk_reward_ratio})"
            )
            return take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {str(e)}")
            return entry_price
            
    def can_open_position(self, current_positions: int) -> bool:
        """
        Check if a new position can be opened based on risk parameters.
        
        Args:
            current_positions: Number of currently open positions
            
        Returns:
            bool: True if new position can be opened
        """
        if current_positions >= self.max_positions:
            self.logger.warning(
                f"Cannot open new position: Maximum positions ({self.max_positions}) reached"
            )
            return False
        return True 