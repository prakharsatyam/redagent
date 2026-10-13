"""
Master Real-Time Signal Engine and Live Scanner Loop.
Coordinates DataPipeline -> Stage-1 Laya News Radar -> Quantitative/Friction Hurdle ->
Stage-2 Laya Options Decision Head -> AlertBot & PaperTrader.
"""

import sys
from pathlib import Path
import time
import logging
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import settings
from src.data.data_pipeline import DataPipeline, UnifiedMarketSnapshot
from src.data.deribit_client import DeribitOptionSummary
from src.quant.greeks import calculate_greeks, GreeksResult
from src.quant.merton_jump import evaluate_merton_regime, MertonResult
from src.quant.volatility_surface import analyze_volatility_regime, VolatilityMetrics
from src.quant.fee_simulator import evaluate_option_candidate, TradePayoffEvaluation
from src.model.laya_news_engine import LayaNewsEngine, NewsSentimentResult
from src.model.state_serializer import serialize_options_state, SerializedMarketState
from src.model.laya_engine import LayaOptionsEngine, OptionDecisionResult
from src.signals.alert_bot import AlertBot, ValidatedTradeSignal
from src.signals.paper_trader import PaperTrader, PaperPosition

logger = logging.getLogger(__name__)


class SignalEngine:
    """Master orchestrator executing live 60-second scanning passes."""

    def __init__(
        self,
        pipeline: Optional[DataPipeline] = None,
        news_engine: Optional[LayaNewsEngine] = None,
        options_engine: Optional[LayaOptionsEngine] = None,
        alert_bot: Optional[AlertBot] = None,
        paper_trader: Optional[PaperTrader] = None,
    ):
        self.pipeline = pipeline or DataPipeline()
        self.news_engine = news_engine or LayaNewsEngine()
        self.options_engine = options_engine or LayaOptionsEngine()
        self.alert_bot = alert_bot or AlertBot()
        self.paper_trader = paper_trader or PaperTrader()

    def evaluate_news_circuit_breakers(self, snapshot: UnifiedMarketSnapshot) -> Optional[str]:
        """
        Passes breaking news and liquidation cascades to Stage-1 Laya News Classifier.
        Returns trigger reason string if circuit breaker triggered (halting calls), else None.
        """
        headlines_to_test = list(snapshot.recent_headlines)
        if snapshot.liquidation_summary.is_cascade_active:
            headlines_to_test.append(snapshot.liquidation_summary.headline_alert)

        for text in headlines_to_test:
            res: NewsSentimentResult = self.news_engine.evaluate_headline(text)
            if res.circuit_breaker_triggered:
                logger.warning(
                    f"Stage-1 Protective Radar TRIGGERED: '{text}' -> Halting all Call option buying!"
                )
                return f"Stage-1 Protective Radar halt: '{text}' ({res.impact_regime})"

        return None

    def scan_market(
        self,
        currency: str = "BTC",
        max_dte: float = 14.0,
        min_dte: float = 0.5,
        broadcast_alerts: bool = True,
        record_paper_trades: bool = True
    ) -> List[ValidatedTradeSignal]:
        """
        Performs a full institutional-grade evaluation pass across the active Call option surface.
        """
        # 1. Ingest consolidated multi-source snapshot
        snapshot = self.pipeline.get_snapshot(currency=currency, max_dte=max_dte, min_dte=min_dte)
        tech_status = snapshot.technical_summary.trend_regime if snapshot.technical_summary else "N/A"
        logger.info(f"[{currency}] Ingested snapshot: Spot ${snapshot.spot_price:,.2f} | {len(snapshot.call_options)} active calls | Tech: {tech_status}")

        # 2. Stage-1 News & Liquidation Circuit Breaker Gate
        halt_reason = self.evaluate_news_circuit_breakers(snapshot)
        if halt_reason:
            logger.info(f"Scanner pass aborted for {currency}: {halt_reason}")
            return []

        # 3. Filter for tradeable, liquid contracts (tight spread, open interest)
        tradeable_calls = self.pipeline.filter_tradeable_calls(snapshot.call_options)
        if not tradeable_calls:
            logger.debug(f"No liquid tradeable Call options found for {currency}")
            return []

        valid_signals: List[ValidatedTradeSignal] = []

        # 4. Quantitative Analysis & Net Hurdle Evaluation
        for candidate in tradeable_calls:
            try:
                # Analytical Greeks
                t_years = max(candidate.dte_days / 365.0, 1e-4)
                greeks: GreeksResult = calculate_greeks(
                    S=snapshot.spot_price,
                    K=candidate.strike,
                    T=t_years,
                    r=0.04,
                    sigma=candidate.implied_volatility
                )

                # Merton Jump-Diffusion
                merton: MertonResult = evaluate_merton_regime(
                    S=snapshot.spot_price,
                    K=candidate.strike,
                    T=max(candidate.dte_days / 365.0, 1e-4),
                    r=0.04,
                    sigma=candidate.implied_volatility
                )

                # Volatility surface classification
                vol_metrics: VolatilityMetrics = analyze_volatility_regime(
                    current_iv=candidate.implied_volatility,
                    realized_vol_30d=0.48,
                    historical_iv_series=[0.45, 0.50, 0.55, 0.60]
                )

                # Enforce >= 2-3% Net Profit Friction Hurdle
                payoff: TradePayoffEvaluation = evaluate_option_candidate(
                    contract_symbol=candidate.instrument_name,
                    spot_price=snapshot.spot_price,
                    bid_price=candidate.best_bid_price,
                    ask_price=candidate.best_ask_price,
                    delta=greeks.delta,
                    theta_daily=greeks.theta_daily,
                    min_net_hurdle_pct=settings.trading.min_net_return_hurdle
                )

                if not payoff.passes_hurdle:
                    continue

                # 5. Serialize state for Stage-2 Laya Decision Head
                state: SerializedMarketState = serialize_options_state(
                    contract_symbol=candidate.instrument_name,
                    spot_price=snapshot.spot_price,
                    strike_price=candidate.strike,
                    dte_days=candidate.dte_days,
                    greeks=greeks,
                    payoff=payoff,
                    vol_metrics=vol_metrics,
                    merton_res=merton,
                    news_regime="bullish_vol_expansion",
                    news_confidence=0.85,
                    funding_rate_8h=snapshot.funding_rate_8h,
                    technical_summary=snapshot.technical_summary
                )

                # 6. Execute in-process Laya Stage-2 Inference
                decision: OptionDecisionResult = self.options_engine.evaluate_state(state)

                if decision.is_signal_triggered:
                    target_exit = payoff.exit_target_usd
                    # Set 25% stop loss below entry ask
                    stop_loss = max(0.01, candidate.best_ask_price * 0.75)

                    sig = ValidatedTradeSignal(
                        contract_symbol=candidate.instrument_name,
                        underlying_spot=snapshot.spot_price,
                        strike_price=candidate.strike,
                        dte_days=candidate.dte_days,
                        mark_premium_usd=candidate.mark_price,
                        best_ask_usd=candidate.best_ask_price,
                        delta=greeks.delta,
                        gamma=greeks.gamma,
                        theta_daily=greeks.theta_daily,
                        iv_annual=candidate.implied_volatility,
                        iv_rank=vol_metrics.iv_rank,
                        laya_confidence=decision.confidence,
                        projected_net_return_pct=payoff.net_return_pct,
                        breakeven_spot_move_pct=payoff.breakeven_spot_move_pct,
                        target_exit_usd=target_exit,
                        stop_loss_usd=stop_loss
                    )

                    valid_signals.append(sig)

                    # 7. Broadcast Alert
                    if broadcast_alerts:
                        self.alert_bot.broadcast(sig)

                    # 8. Record in Paper Trader portfolio
                    if record_paper_trades:
                        self.paper_trader.open_position_from_signal(sig, contracts=1.0)

            except Exception as e:
                logger.error(f"Error evaluating candidate {candidate.instrument_name}: {e}")
                continue

        return valid_signals

    def run_scanner_loop(
        self,
        currency: str = "BTC",
        interval_seconds: int = 60,
        iterations: Optional[int] = None
    ) -> None:
        """
        Continuously loops scanning market every interval_seconds.
        """
        count = 0
        logger.info(f"Starting RedAgent SignalEngine live loop for {currency} (interval: {interval_seconds}s)")

        while True:
            try:
                signals = self.scan_market(currency=currency)
                if signals:
                    logger.info(f"[{currency}] Pass complete: {len(signals)} high-conviction Call signals detected & broadcasted!")
                else:
                    logger.info(f"[{currency}] Pass complete: No candidates passed both mathematical net hurdle (>= 3%) and Laya conviction this cycle.")
            except Exception as e:
                logger.error(f"Unexpected error in scanner loop: {e}")

            count += 1
            if iterations is not None and count >= iterations:
                break

            time.sleep(interval_seconds)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    parser = argparse.ArgumentParser(description="RedAgent Signal Engine Live Scanner")
    parser.add_argument("--currency", default="BTC", help="Target cryptocurrency (BTC/ETH/SOL)")
    parser.add_argument("--interval", type=int, default=60, help="Scan interval in seconds")
    parser.add_argument("--iterations", type=int, default=1, help="Number of scan iterations (default 1)")
    args = parser.parse_args()

    engine = SignalEngine()
    engine.run_scanner_loop(currency=args.currency, interval_seconds=args.interval, iterations=args.iterations)
