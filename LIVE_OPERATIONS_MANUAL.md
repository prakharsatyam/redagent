# RedAgent: Production Readiness & Live Operations Runbook

---

## 1. Comprehensive System Inventory: What Has Been Done

RedAgent is an institutional-grade crypto Call options decision engine. Below is the complete audit of what is built, tested, and ready in the repository:

| Subsystem | Core Module | Current Capabilities & Enhancements |
| :--- | :--- | :--- |
| **Multi-Asset Ingestion** | [`src/data/deribit_client.py`](file:///Users/prakharsatyam/codes/redagent/src/data/deribit_client.py) | • **BTC & ETH:** Inverse coin-margined options quotation conversion ($Quote \times Spot$).<br>• **SOL:** Linear USDC-settled contracts under `currency=USDC` (`SOL_USDC-*`).<br>• Public REST endpoints (zero private API keys required for market scanning). |
| **Spot & Sentiment** | [`src/data/binance_client.py`](file:///Users/prakharsatyam/codes/redagent/src/data/binance_client.py)<br>[`src/data/liquidation_feed.py`](file:///Users/prakharsatyam/codes/redagent/src/data/liquidation_feed.py) | • Real-time spot price, 8-hour perpetual funding rate, and open interest for BTC, ETH, SOL.<br>• 15-minute rolling liquidation cascade detector (`HEAVY_LONG_CASCADE` vs `SHORT_SQUEEZE`).<br>• 1H/4H OHLCV candlestick historical klines fetcher. |
| **Parametric Chart Vision** | [`src/quant/technical_indicators.py`](file:///Users/prakharsatyam/codes/redagent/src/quant/technical_indicators.py) | • 20 EMA, 50 EMA, 200 SMA trend classification (`BULLISH_UPTREND`, etc.).<br>• RSI(14) with Wilder smoothing and zero-loss handling.<br>• MACD (12, 26, 9) histogram and crossover bias.<br>• Bollinger Bands (20, 2.0 std dev) with bandwidth volatility squeeze detection.<br>• Rolling 24h Support / Resistance levels and Volume-Weighted Average Price (VWAP). |
| **News & Circuit Breaker** | [`src/data/news_streamer.py`](file:///Users/prakharsatyam/codes/redagent/src/data/news_streamer.py)<br>[`src/model/laya_news_engine.py`](file:///Users/prakharsatyam/codes/redagent/src/model/laya_news_engine.py) | • Live CryptoPanic news aggregator with hot keyword filtering.<br>• Sub-15ms Stage-1 Laya News Radar acting as an automated circuit breaker (halts call purchases on bearish dumps/exploits). |
| **Quantitative Mathematics** | [`src/quant/greeks.py`](file:///Users/prakharsatyam/codes/redagent/src/quant/greeks.py)<br>[`src/quant/merton_jump.py`](file:///Users/prakharsatyam/codes/redagent/src/quant/merton_jump.py)<br>[`src/quant/volatility_surface.py`](file:///Users/prakharsatyam/codes/redagent/src/quant/volatility_surface.py) | • Analytical Black-Scholes-Merton Greeks ($\Delta, \Gamma, \Theta, \nu, \rho$) with Brent IV root solver.<br>• Merton (1976) Jump-Diffusion pricing & Corrado-Su non-Gaussian skewness adjustment.<br>• 90-day IV Rank, 30-day Realized Vol, Volatility Risk Premium (VRP), and SABR smile. |
| **Friction & Hurdle Gate** | [`config/fees.py`](file:///Users/prakharsatyam/codes/redagent/config/fees.py)<br>[`src/quant/fee_simulator.py`](file:///Users/prakharsatyam/codes/redagent/src/quant/fee_simulator.py) | • Deribit fee schedule (0.03% spot capped at 12.5% premium) + bid-ask spread crossing.<br>• Indian Section 115BBH (31.2% tax) and 194S (1% TDS) drag calculations.<br>• **Enforces Net Hurdle:** Mathematically rejects any option with projected net profit $< 3.0\%$. |
| **Context Fusion Engine** | [`src/model/state_serializer.py`](file:///Users/prakharsatyam/codes/redagent/src/model/state_serializer.py)<br>[`src/model/laya_engine.py`](file:///Users/prakharsatyam/codes/redagent/src/model/laya_engine.py) | • Fuses quant Greeks, technical indicators, news sentiment, and fee payoffs into a typed System 1 state document.<br>• In-process Laya runtime delivering typed classifications (`should_buy_call`, `expected_net_return`, `risk_regime`) in sub-25ms. |
| **Dataset Generation** | [`scripts/fetch_historical_options.py`](file:///Users/prakharsatyam/codes/redagent/scripts/fetch_historical_options.py)<br>[`src/model/dataset_generator.py`](file:///Users/prakharsatyam/codes/redagent/src/model/dataset_generator.py) | • Downloads live contracts + generates 3,000–10,000+ stochastic GBM & Merton jump ticks.<br>• Exports walk-forward `train.jsonl` and `val.jsonl` with hard negative mining on bull traps. |
| **Execution & Portfolio** | [`src/signals/alert_bot.py`](file:///Users/prakharsatyam/codes/redagent/src/signals/alert_bot.py)<br>[`src/signals/paper_trader.py`](file:///Users/prakharsatyam/codes/redagent/src/signals/paper_trader.py)<br>[`src/signals/signal_engine.py`](file:///Users/prakharsatyam/codes/redagent/src/signals/signal_engine.py) | • Professional Markdown alert dispatch to Telegram/Discord.<br>• Virtual paper ledger tracking open positions, target exits (+20%), and stop losses (-25%) in [`data/paper_portfolio.json`](file:///Users/prakharsatyam/codes/redagent/data/paper_portfolio.json).<br>• Master 60-second real-time scanning loop. |
| **Test Suite** | `tests/` | • **74 unit tests passing 100%** across the entire pipeline. |

---

## 2. Step-by-Step Production Roadmap

```mermaid
flowchart TD
    S0["Step 0: Verification & Keys Setup<br/>(tests/ pass + keys.env)"] --> S1["Step 1: Market Data & Indicators Inspection<br/>(Live Binance & Deribit checks)"]
    S1 --> S2["Step 2: Model Weights Setup<br/>(Colab fine-tune or built-in heuristic)"]
    S2 --> S3["Step 3: Alert Channel Verification<br/>(Telegram test broadcast)"]
    S3 --> S4["Step 4: Single-Pass Scanner Dry Run<br/>(Verify Greeks, Hurdle, Payoff)"]
    S4 --> S5["Step 5: Launch Live Scanner Daemon<br/>(Automated paper trading & real-time alerts)"]
```

---

## 3. Step 0: Verification & Environment Setup

### 3.1 Run Full Test Suite Sanity Check
Before doing anything, ensure the local virtual environment passes all tests:
```bash
source .venv/bin/activate
pytest tests/ -v
```
*Expected Result:* `74 passed in ~2.5s`.

### 3.2 Configure API Credentials in `keys.env`
Edit [`keys.env`](file:///Users/prakharsatyam/codes/redagent/keys.env) with your credentials:
```bash
# Telegram Alert Bot (Recommended for real-time alerts)
TELEGRAM_BOT_TOKEN="1234567890:ABCdefGHIjklMNOpqrsTUVwxyz"
TELEGRAM_CHAT_ID="123456789"

# Breaking News (Optional - falls back to simulated feed if empty)
CRYPTOPANIC_API_KEY="your_cryptopanic_token"

# Deribit & Binance (Optional - public endpoints are used for scanning)
DERIBIT_CLIENT_ID=""
DERIBIT_CLIENT_SECRET=""
BINANCE_API_KEY=""
BINANCE_API_SECRET=""
```

---

## 4. Step 1: Live Market & Technical Data Inspection

Run this one-liner to verify that Binance spot prices, funding rates, and 1H technical indicators are streaming correctly:

```bash
.venv/bin/python -c '
from src.data.binance_client import BinanceClient
from src.quant.technical_indicators import compute_technical_indicators

bc = BinanceClient()
for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
    df = bc.get_klines(sym, interval="1h", limit=50)
    summary = compute_technical_indicators(df, symbol=sym, interval="1h")
    print(f"\n--- {sym} ---")
    print(summary.to_serialized_text())
'
```
*Inspection Checklist:*
- [ ] Spot price matches current market.
- [ ] 20 EMA and 50 EMA calculated.
- [ ] RSI(14) between 0 and 100.
- [ ] BB width and Volatility Compression status displayed.

---

## 5. Step 2: Model Weights & Fine-Tuning Setup

RedAgent supports two operational modes:
- **Mode A (Immediate / Zero-Shot Heuristic):** If no fine-tuned checkpoint exists in `models/laya_options_agent/`, RedAgent automatically uses calibrated quantitative heuristics (BSM + Merton + Volatility Surface + Hurdle Gate). **You can start immediately without training.**
- **Mode B (Fine-Tuned Laya Weights):** For deep System 1 ModernBERT neural classification.

### To Train the Laya Head (Mode B):
1. **Generate 3,000+ Training Ticks:**
   ```bash
   .venv/bin/python scripts/fetch_historical_options.py --currency ALL --samples 3000
   ```
2. **Process into JSONL Training Sets:**
   ```bash
   .venv/bin/python src/model/dataset_generator.py \
     --input_parquet data/raw/deribit_btc.parquet \
     --train_out data/processed/train.jsonl \
     --val_out data/processed/val.jsonl
   ```
3. **Train in Google Colab (5–10 mins on free GPU):**
   - Open [`laya.ipynb`](file:///Users/prakharsatyam/codes/redagent/laya.ipynb) in Google Colab.
   - Sync data or upload `data/processed/train.jsonl`.
   - Run the notebook to train the decision head.
   - Download the checkpoint folder into [`models/laya_options_agent/`](file:///Users/prakharsatyam/codes/redagent/models/laya_options_agent/).

---

## 6. Step 3: Telegram Alert Channel Verification

Verify that your Telegram Bot token and Chat ID are working:

```bash
.venv/bin/python -c '
from src.signals.alert_bot import AlertBot, ValidatedTradeSignal

bot = AlertBot()
if not bot.bot_token or not bot.chat_id:
    print("Telegram credentials not configured in keys.env (Alerts will be logged to console only).")
else:
    test_sig = ValidatedTradeSignal(
        contract_symbol="BTC-TEST-CALL",
        underlying_spot=83500.0,
        strike_price=85000.0,
        dte_days=3.0,
        mark_premium_usd=1200.0,
        best_ask_usd=1220.0,
        delta=0.42,
        gamma=0.000035,
        theta_daily=-110.0,
        iv_annual=0.52,
        iv_rank=34.0,
        laya_confidence=0.88,
        projected_net_return_pct=0.185,
        breakeven_spot_move_pct=1.45,
        target_exit_usd=1464.0,
        stop_loss_usd=915.0
    )
    ok = bot.send_telegram(bot.format_alert(test_sig))
    print("Telegram test message sent:", "SUCCESS" if ok else "FAILED (Check bot token & chat ID)")
'
```

---

## 7. Step 4: Single-Pass Scanner Dry Run

Perform a single scan iteration to inspect live options evaluation:

```bash
# Single dry-run for Bitcoin:
.venv/bin/python src/signals/signal_engine.py --currency BTC --iterations 1

# Single dry-run for Solana:
.venv/bin/python src/signals/signal_engine.py --currency SOL --iterations 1

# Single dry-run for Ethereum:
.venv/bin/python src/signals/signal_engine.py --currency ETH --iterations 1
```

*What the engine does during this run:*
1. Ingests Deribit active Call options chain.
2. Ingests Binance spot price, 8h funding rate, and 1H candlestick indicators.
3. Checks Stage-1 News circuit breaker (aborts if a crash/exploit is detected).
4. Filters liquid contracts (spread $< 12\%$, open interest $> 0$).
5. Computes analytical Greeks ($\Delta, \Gamma, \Theta$) and Merton Jump-Diffusion fair price.
6. Runs Fee & Hurdle simulator: calculates round-trip maker/taker fees, spread crossing, and tax drag.
7. Discards any setup that fails the $\ge 3\%$ net return constraint.
8. Passes candidates to Laya System 1 decision head.
9. Logs high-conviction signals and registers virtual positions in [`data/paper_portfolio.json`](file:///Users/prakharsatyam/codes/redagent/data/paper_portfolio.json).

---

## 8. Step 5: Going Live — Running the Scanner Daemon

### 8.1 Interactive Foreground Execution
```bash
# Runs every 60 seconds indefinitely:
.venv/bin/python src/signals/signal_engine.py --currency BTC --interval 60 --iterations 0
```

### 8.2 Background Production Execution (nohup / screen / tmux)

#### Using `nohup`:
```bash
nohup .venv/bin/python src/signals/signal_engine.py --currency BTC --interval 60 --iterations 0 > scanner_btc.log 2>&1 &
echo $! > scanner_btc.pid
```

#### Multi-Asset Background Execution:
```bash
# Start BTC scanner:
nohup .venv/bin/python src/signals/signal_engine.py --currency BTC --interval 60 --iterations 0 > logs/scanner_btc.log 2>&1 &

# Start SOL scanner:
nohup .venv/bin/python src/signals/signal_engine.py --currency SOL --interval 60 --iterations 0 > logs/scanner_sol.log 2>&1 &

# Start ETH scanner:
nohup .venv/bin/python src/signals/signal_engine.py --currency ETH --interval 60 --iterations 0 > logs/scanner_eth.log 2>&1 &
```

---

## 9. Live Monitoring & Portfolio Inspections

### 9.1 Monitor Live Logs
```bash
tail -f scanner_btc.log
```

### 9.2 Inspect Virtual Paper Portfolio Ledger
Every trade opened by RedAgent is recorded with exact entry prices, fees, target exits ($+20\%$), and stop losses ($-25\%$).

```bash
# Check open and closed positions:
.venv/bin/python -c '
from src.signals.paper_trader import PaperTrader
pt = PaperTrader()
summary = pt.get_summary()
print(f"Total Trades: {summary.total_trades}")
print(f"Open Positions: {summary.open_positions_count}")
print(f"Closed Positions: {summary.closed_positions_count} (Wins: {summary.win_count}, Losses: {summary.loss_count})")
print(f"Win Rate: {summary.win_rate:.1%}")
print(f"Total Realized Net PnL: ${summary.total_realized_pnl_usd:+,.2f}")
print(f"Total Fees Paid: ${summary.total_fees_paid_usd:,.2f}")
'
```

### 9.3 Stopping the Daemon
```bash
kill $(cat scanner_btc.pid)
```
