"""
Paper Trader and Virtual Portfolio PnL Tracker.
Simulates live options execution, tracks positions, checks stop-losses/targets,
and calculates real-world net returns without financial or tax risk.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from config.settings import settings
from config.fees import calculate_deribit_fee
from src.signals.alert_bot import ValidatedTradeSignal


class PaperPosition(BaseModel):
    """An open simulated options trade position."""
    position_id: str
    contract_symbol: str
    entry_timestamp: str
    entry_ask_usd: float
    contracts_count: float = 1.0
    underlying_entry_spot: float
    target_exit_usd: float
    stop_loss_usd: float
    entry_fee_usd: float
    current_bid_usd: float
    status: str = Field(default="OPEN", description="'OPEN', 'CLOSED_TARGET', 'CLOSED_STOP'")
    exit_timestamp: Optional[str] = None
    exit_price_usd: Optional[float] = None
    exit_fee_usd: Optional[float] = None
    realized_pnl_usd: Optional[float] = None
    realized_return_pct: Optional[float] = None


class PaperPortfolioSummary(BaseModel):
    """Summary metrics of the virtual paper portfolio."""
    total_trades: int
    open_positions_count: int
    closed_positions_count: int
    win_count: int
    loss_count: int
    win_rate: float
    total_realized_pnl_usd: float
    total_fees_paid_usd: float


class PaperTrader:
    """Manages virtual positions, tracking, and JSON persistence."""

    def __init__(self, ledger_file: Optional[Path] = None):
        self.ledger_file = ledger_file or (settings.data_dir / "paper_portfolio.json")
        self.positions: Dict[str, PaperPosition] = {}
        self._load_ledger()

    def _load_ledger(self):
        """Loads positions ledger from JSON if exists."""
        if self.ledger_file.exists():
            try:
                with open(self.ledger_file, "r") as f:
                    data = json.load(f)
                    for pid, pdata in data.items():
                        self.positions[pid] = PaperPosition(**pdata)
            except Exception:
                self.positions = {}

    def _save_ledger(self):
        """Saves positions to JSON file."""
        self.ledger_file.parent.mkdir(parents=True, exist_ok=True)
        data = {pid: pos.model_dump() for pid, pos in self.positions.items()}
        with open(self.ledger_file, "w") as f:
            json.dump(data, f, indent=2)

    def open_position_from_signal(self, sig: ValidatedTradeSignal, contracts: float = 1.0) -> PaperPosition:
        """Opens a simulated long call position based on a validated signal."""
        pid = f"{sig.contract_symbol}_{int(datetime.now(timezone.utc).timestamp())}"
        entry_fee = calculate_deribit_fee(sig.underlying_spot, sig.best_ask_usd, is_maker=False) * contracts

        pos = PaperPosition(
            position_id=pid,
            contract_symbol=sig.contract_symbol,
            entry_timestamp=datetime.now(timezone.utc).isoformat(),
            entry_ask_usd=sig.best_ask_usd,
            contracts_count=contracts,
            underlying_entry_spot=sig.underlying_spot,
            target_exit_usd=sig.target_exit_usd,
            stop_loss_usd=sig.stop_loss_usd,
            entry_fee_usd=entry_fee,
            current_bid_usd=sig.mark_premium_usd,
            status="OPEN"
        )
        self.positions[pid] = pos
        self._save_ledger()
        return pos

    def update_position_price(self, position_id: str, current_bid: float, current_spot: float) -> Optional[PaperPosition]:
        """
        Updates current market price and automatically exits if target or stop-loss hit.
        """
        if position_id not in self.positions:
            return None

        pos = self.positions[position_id]
        if pos.status != "OPEN":
            return pos

        pos.current_bid_usd = current_bid

        # Check Target Hit
        if current_bid >= pos.target_exit_usd:
            self._close_position(pos, exit_price=current_bid, spot_price=current_spot, status="CLOSED_TARGET")
        # Check Stop Loss Hit
        elif current_bid <= pos.stop_loss_usd:
            self._close_position(pos, exit_price=current_bid, spot_price=current_spot, status="CLOSED_STOP")

        self._save_ledger()
        return pos

    def _close_position(self, pos: PaperPosition, exit_price: float, spot_price: float, status: str):
        """Calculates realized net PnL and closes position."""
        exit_fee = calculate_deribit_fee(spot_price, exit_price, is_maker=True) * pos.contracts_count
        gross_pnl = (exit_price - pos.entry_ask_usd) * pos.contracts_count
        net_pnl = gross_pnl - pos.entry_fee_usd - exit_fee
        ret_pct = (net_pnl / (pos.entry_ask_usd * pos.contracts_count)) if pos.entry_ask_usd > 0 else 0.0

        pos.status = status
        pos.exit_timestamp = datetime.now(timezone.utc).isoformat()
        pos.exit_price_usd = exit_price
        pos.exit_fee_usd = exit_fee
        pos.realized_pnl_usd = net_pnl
        pos.realized_return_pct = ret_pct

    def get_summary(self) -> PaperPortfolioSummary:
        """Calculates win rate, realized PnL, and fee metrics."""
        total = len(self.positions)
        open_count = sum(1 for p in self.positions.values() if p.status == "OPEN")
        closed_positions = [p for p in self.positions.values() if p.status != "OPEN"]
        closed_count = len(closed_positions)

        wins = sum(1 for p in closed_positions if (p.realized_pnl_usd or 0) > 0)
        losses = closed_count - wins
        win_rate = (wins / closed_count) if closed_count > 0 else 0.0
        total_pnl = sum((p.realized_pnl_usd or 0) for p in closed_positions)
        total_fees = sum((p.entry_fee_usd + (p.exit_fee_usd or 0)) for p in self.positions.values())

        return PaperPortfolioSummary(
            total_trades=total,
            open_positions_count=open_count,
            closed_positions_count=closed_count,
            win_count=wins,
            loss_count=losses,
            win_rate=win_rate,
            total_realized_pnl_usd=total_pnl,
            total_fees_paid_usd=total_fees
        )
