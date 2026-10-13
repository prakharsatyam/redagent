"""
Stage-2 Walk-Forward Dataset Generator and Hard Negative Miner.
Processes historical Deribit options ticks, simulates real-world Indian friction/fees,
performs hard negative mining on bull-traps, and exports JSONL datasets for Colab fine-tuning.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import settings
from src.quant.greeks import calculate_greeks, GreeksResult
from src.quant.merton_jump import evaluate_merton_regime, MertonResult
from src.quant.volatility_surface import analyze_volatility_regime, VolatilityMetrics
from src.quant.fee_simulator import evaluate_option_candidate, TradePayoffEvaluation
from src.model.state_serializer import serialize_options_state, OPTIONS_DECISION_QUESTIONS


class DatasetGenerator:
    """Generates walk-forward JSONL datasets formatted for Laya System 1 decision training."""

    def __init__(self, min_net_hurdle: float = 0.03, train_split_ratio: float = 0.80):
        self.min_net_hurdle = min_net_hurdle
        self.train_split_ratio = train_split_ratio

    def process_records_from_df(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Replays options ticks from dataframe, applies friction calculus,
        and generates Laya question-answer training tuples.
        """
        # Ensure chronological ordering
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        dataset_rows = []

        for _, row in df.iterrows():
            spot = float(row.get("spot_price", 65000.0))
            strike = float(row.get("strike", 65000.0))
            dte = float(row.get("dte_days", 5.0))
            bid = float(row.get("bid_price", 1000.0))
            ask = float(row.get("ask_price", 1040.0))
            iv = float(row.get("iv", 0.50))
            funding = float(row.get("funding_rate_8h", 0.0001))
            forward_spot_move = float(row.get("forward_spot_move_pct", 0.025))
            symbol = str(row.get("symbol", f"BTC-{int(strike)}-C"))

            t_years = max(dte / 365.0, 1e-4)

            # 1. Analytical Greeks
            greeks: GreeksResult = calculate_greeks(
                S=spot, K=strike, T=t_years, r=0.04, sigma=iv
            )

            # 2. Merton Jump Regime
            merton: MertonResult = evaluate_merton_regime(
                S=spot, K=strike, T=t_years, r=0.04, sigma=iv
            )

            # 3. Volatility Surface
            vol_metrics: VolatilityMetrics = analyze_volatility_regime(
                current_iv=iv,
                realized_vol_30d=0.45,
                historical_iv_series=[0.40, 0.45, 0.50, 0.55]
            )

            # 4. Friction & Payoff Simulation
            payoff: TradePayoffEvaluation = evaluate_option_candidate(
                contract_symbol=symbol,
                spot_price=spot,
                bid_price=bid,
                ask_price=ask,
                delta=greeks.delta,
                theta_daily=greeks.theta_daily,
                target_spot_move_pct=forward_spot_move,
                min_net_hurdle_pct=self.min_net_hurdle
            )

            # 5. Label Assignment
            won_trade = (
                payoff.passes_hurdle and
                forward_spot_move >= payoff.breakeven_spot_move_pct and
                payoff.net_return_pct >= self.min_net_hurdle
            )

            should_buy_choice = "yes" if won_trade else "no"

            # Return Bracket
            if payoff.net_return_pct < 0.0:
                ret_bracket = "loss_or_negative"
            elif payoff.net_return_pct < 0.02:
                ret_bracket = "breakeven_to_2pct"
            elif payoff.net_return_pct <= 0.10:
                ret_bracket = "target_3pct_to_10pct"
            else:
                ret_bracket = "explosive_above_10pct"

            # Risk Regime
            if dte <= 2.0:
                risk_reg = "high_theta_decay"
            elif iv >= 0.65:
                risk_reg = "iv_crush_risk"
            elif iv <= 0.45:
                risk_reg = "favorable_low_vol_expansion"
            else:
                risk_reg = "trending_spot_momentum"

            # 6. Serialize State
            serialized = serialize_options_state(
                contract_symbol=symbol,
                spot_price=spot,
                strike_price=strike,
                dte_days=dte,
                greeks=greeks,
                payoff=payoff,
                vol_metrics=vol_metrics,
                merton_res=merton,
                news_regime="neutral_noise",
                news_confidence=0.70,
                funding_rate_8h=funding
            )

            record = {
                "state": serialized.state_text,
                "questions": OPTIONS_DECISION_QUESTIONS,
                "labels": {
                    "should_buy_call": should_buy_choice,
                    "expected_net_return": ret_bracket,
                    "risk_regime": risk_reg
                }
            }
            dataset_rows.append(record)

            # Hard Negative Mining: If spot went up, but contract lost money (Bull Trap),
            # duplicate this record so Laya strongly learns to avoid theta/spread traps!
            if forward_spot_move > 0.01 and not won_trade:
                dataset_rows.append(record)

        return dataset_rows

    def split_walk_forward(
        self, records: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Chronological walk-forward split (train 80%, val 20%)."""
        split_idx = int(len(records) * self.train_split_ratio)
        train_records = records[:split_idx]
        val_records = records[split_idx:]
        return train_records, val_records

    def save_jsonl(self, records: List[Dict[str, Any]], output_path: Path) -> int:
        """Writes records to a JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row) + "\n")
        return len(records)

    def generate_synthetic_historical_df(self, n_samples: int = 120) -> pd.DataFrame:
        """Generates realistic synthetic options tick history for testing and bootstrap."""
        np.random.seed(42)
        base_spot = 65000.0
        records = []
        for i in range(n_samples):
            spot = base_spot + np.random.normal(0, 1500.0)
            strike_offset = np.random.choice([-2000, -1000, 0, 1000, 2000, 4000])
            strike = round(spot + strike_offset, -2)
            dte = np.random.uniform(1.0, 12.0)
            iv = np.random.uniform(0.40, 0.70)
            spread_pct = np.random.uniform(0.015, 0.06)

            # Estimate theoretical mid
            mid = max(50.0, (spot - strike) * 0.5 + spot * iv * np.sqrt(dte / 365.0) * 0.4)
            bid = mid * (1.0 - spread_pct / 2.0)
            ask = mid * (1.0 + spread_pct / 2.0)

            # Forward spot move after 24 hours
            forward_move = np.random.normal(0.005, 0.03)
            funding = np.random.uniform(-0.0002, 0.0004)

            records.append({
                "timestamp": f"2026-10-{1 + (i % 28):02d}T12:00:00Z",
                "symbol": f"BTC-{int(strike)}-C",
                "spot_price": spot,
                "strike": strike,
                "dte_days": dte,
                "bid_price": bid,
                "ask_price": ask,
                "iv": iv,
                "funding_rate_8h": funding,
                "forward_spot_move_pct": forward_move
            })

        return pd.DataFrame(records)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage-2 Dataset Generator for Laya Decision Training")
    parser.add_argument("--input_parquet", default=None, help="Input parquet path or directory (e.g. data/raw/deribit_btc.parquet)")
    parser.add_argument("--samples", type=int, default=3000, help="Number of synthetic samples if no parquet provided (default 3000)")
    parser.add_argument("--train_out", default="data/processed/train.jsonl", help="Output path for training JSONL")
    parser.add_argument("--val_out", default="data/processed/val.jsonl", help="Output path for validation JSONL")
    args = parser.parse_args()

    gen = DatasetGenerator()
    df = None

    if args.input_parquet:
        p_path = Path(args.input_parquet)
        if p_path.is_file():
            df = pd.read_parquet(p_path)
            print(f"Loaded {len(df)} records from {p_path}")
        elif p_path.is_dir():
            dfs = [pd.read_parquet(f) for f in p_path.glob("*.parquet")]
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                print(f"Loaded {len(df)} records across {len(dfs)} parquet files in {p_path}")

    # Fallback to default raw parquet or generate high-sample synthetic dataframe
    if df is None:
        default_btc = Path("data/raw/deribit_btc.parquet")
        if default_btc.exists():
            df = pd.read_parquet(default_btc)
            print(f"Auto-loaded {len(df)} records from {default_btc}")
        else:
            print(f"Generating {args.samples} calibrated synthetic samples...")
            df = gen.generate_synthetic_historical_df(args.samples)

    records = gen.process_records_from_df(df)
    train_rec, val_rec = gen.split_walk_forward(records)

    gen.save_jsonl(train_rec, Path(args.train_out))
    gen.save_jsonl(val_rec, Path(args.val_out))
    print(f"Generated {len(train_rec)} training samples -> {args.train_out}")
    print(f"Generated {len(val_rec)} validation samples -> {args.val_out}")
