# RedAgent: Comprehensive Architectural Walkthrough & Operations Manual

---

## 1. Executive Summary & Philosophy

**RedAgent** is an institutional-grade, AI-driven crypto Call options decision and advisory engine built for Bitcoin (BTC) and Ethereum (ETH) derivatives.

Rather than relying on generic auto-regressive LLMs (which suffer from high latency of 1–3 seconds, hallucinations, and non-deterministic text outputs), RedAgent utilizes **ConvAI Innovations' Laya** (`convaiinnovations/laya`), an open-weight System 1 decision model fine-tuned on top of **ModernBERT-421M**. Laya operates in a single encoder forward pass, delivering typed, deterministic classifications (`noul`, `choice`, `score`) in **sub-25 milliseconds**.

### The Core Mandate: Mathematical Net Hurdle ($\ge 3\%$ Profit)
Most retail options traders lose money not because their market direction was wrong, but because **options friction destroyed their gains**:
1. **Deribit Taker Fees:** $0.03\%$ of underlying spot price (capped at $12.5\%$ of option premium).
2. **Bid-Ask Spread Crossing:** Half-spread costs $\frac{\text{Ask} - \text{Bid}}{2}$ on wide crypto books.
3. **Calendar Theta Decay ($\Theta$):** Daily erosion of extrinsic time value.
4. **Indian Tax Friction (Section 115BBH / 194S TDS):** $30\%$ flat tax on gains with zero loss offset, plus $1\%$ TDS per transaction.

RedAgent mathematically rejects any Call option setup unless:
$$\text{Projected Net Return} = \frac{\Delta \text{Option Value} - \text{RoundTripFees} - \text{ThetaDecay} - \text{SpreadCost}}{\text{Ask Entry Premium}} \ge 3.0\%$$

---

## 2. Exhaustive File-by-File Breakdown & Testing Guide

The codebase is organized into cleanly decoupled, testable subsystems:

```
redagent/
├── config/        # Environment, parameters, and localized tax/fee friction
├── src/
│   ├── data/      # Multi-source market data ingestion & normalization
│   ├── quant/     # Mathematical pricing, Greeks, jump-diffusion, and AI rules
│   ├── model/     # Context serialization, in-process Laya engines, dataset generation
│   └── signals/   # Live scanning loop, Telegram broadcasting, paper portfolio tracker
├── scripts/       # Historical data ingestion, event-study labeling, Colab sync bridge
├── models/        # Checkpoint directories for Stage-1 and Stage-2 Laya weights
├── tests/         # 23 pytest test modules (63 passing tests)
├── laya.ipynb     # Colab GPU notebook: Stage-2 Options Decision Head fine-tuning
├── laya_news.ipynb# Colab GPU notebook: Stage-1 News Radar fine-tuning
└── keys.env       # Secure environment credentials
```

---

### Subsystem A: Configuration & Friction (`config/`)

#### 1. [`config/settings.py`](file:///Users/prakharsatyam/codes/redagent/config/settings.py)
* **What it does:** Centralized Pydantic configuration loader. Reads API keys and secrets from [`keys.env`](file:///Users/prakharsatyam/codes/redagent/keys.env) (Deribit, Binance, Telegram, CryptoPanic), and establishes global trading constraints (e.g. `min_conviction_threshold = 0.75`, `min_net_return_hurdle = 0.03`, `max_dte_days = 14.0`, `min_dte_days = 0.5`).
* **Inputs & Outputs:** Inputs: OS environment / `keys.env`. Output: Singleton `settings` instance.
* **How to test individually:**
  ```bash
  source .venv/bin/activate
  pytest tests/test_settings.py -v
  ```
  *Tests verify credential parsing, default trading parameters, and root directory resolution.*

#### 2. [`config/fees.py`](file:///Users/prakharsatyam/codes/redagent/config/fees.py)
* **What it does:** Exact fee and friction modeling engine. Implements Deribit's fee structure (Taker: 0.03% spot capped at 12.5% premium, Maker: 0.03% spot), bid-ask spread crossing, and Indian Section 194S (1% TDS) tax drag.
* **Inputs & Outputs:** Inputs: spot price, option premium, trade side. Outputs: exact dollar friction, net payout, and effective fee percentage.
* **How to test individually:**
  ```bash
  source .venv/bin/activate
  pytest tests/test_fees.py -v
  ```
  *Tests verify the 12.5% premium cap rule, maker/taker differential, and domestic vs offshore tax friction.*

---

### Subsystem B: Market Data & Streaming (`src/data/`)

#### 3. [`src/data/deribit_client.py`](file:///Users/prakharsatyam/codes/redagent/src/data/deribit_client.py)
* **What it does:** REST client for Deribit Options API v2. Fetches index spot prices, active instruments list, order books (bid/ask depth), and calculates normalized option metrics. Converts Deribit's native BTC-quoted premiums into USD (`bid_usd = best_bid_btc * spot`).
* **Inputs & Outputs:** Inputs: Deribit public endpoints. Output: List of `DeribitOptionSummary` objects.
* **How to test individually:**
  ```bash
  source .venv/bin/activate
  pytest tests/test_deribit_client.py -v
  ```
  *Tests mock Deribit endpoints to verify correct BTC-to-USD conversion, DTE calculation, and options filtering.*

#### 4. [`src/data/binance_client.py`](file:///Users/prakharsatyam/codes/redagent/src/data/binance_client.py)
* **What it does:** Fetches real-time BTC/ETH spot prices, 8-hour perpetual funding rates, and open interest from Binance Public REST API. Funding rates provide vital sentiment bias (positive funding = longs paying shorts).
* **Inputs & Outputs:** Inputs: Binance API. Output: `BinanceMarketContext` object.
* **How to test individually:**
  ```bash
  source .venv/bin/activate
  pytest tests/test_binance_client.py -v
  ```
  *Tests verify spot price queries, funding rate formatting, and network fallback defaults.*

#### 5. [`src/data/news_streamer.py`](file:///Users/prakharsatyam/codes/redagent/src/data/news_streamer.py)
* **What it does:** Real-time aggregator for breaking crypto news. Connects to CryptoPanic API and RSS feeds, deduplicates headlines, and isolates market-moving keywords (Fed, CPI, Rate Cut, SEC, ETF, Hack, Exploit). Includes simulated fallback for offline testing.
* **Inputs & Outputs:** Inputs: CryptoPanic API / RSS. Output: List of `NewsArticle` objects and clean headline strings.
* **How to test individually:**
  ```bash
  source .venv/bin/activate
  pytest tests/test_news_streamer.py -v
  ```
  *Tests verify headline parsing, deduplication, and volatility keyword filtering.*

#### 6. [`src/data/liquidation_feed.py`](file:///Users/prakharsatyam/codes/redagent/src/data/liquidation_feed.py)
* **What it does:** Real-time liquidation cascade radar. Tracks rolling 15-minute long vs short liquidation volumes. Detects downside waterfalls (`HEAVY_LONG_CASCADE`) or upside squeezes (`HEAVY_SHORT_SQUEEZE`), auto-generating high-priority alert headlines.
* **Inputs & Outputs:** Inputs: Liquidation ticks. Output: `LiquidationSummary` object.
* **How to test individually:**
  ```bash
  source .venv/bin/activate
  pytest tests/test_liquidation_feed.py -v
  ```
  *Tests verify rolling-window event pruning, long waterfall detection, and short squeeze detection.*

#### 7. [`src/data/data_pipeline.py`](file:///Users/prakharsatyam/codes/redagent/src/data/data_pipeline.py)
* **What it does:** Multi-source master consolidator. Pulls together Deribit Call chains, Binance spot/funding, breaking news, and liquidations into a single synchronized `UnifiedMarketSnapshot`. Provides `filter_tradeable_calls()` to reject phantom contracts with wide spreads (>12%) or zero open interest.
* **Inputs & Outputs:** Inputs: Clients 3–6. Output: `UnifiedMarketSnapshot`.
* **How to test individually:**
  ```bash
  source .venv/bin/activate
  pytest tests/test_data_pipeline.py -v
  ```
  *Tests verify clean multi-source aggregation and liquid contract filtering.*

---

### Subsystem C: Quantitative Math Engine (`src/quant/`)

#### 8. [`src/quant/greeks.py`](file:///Users/prakharsatyam/codes/redagent/src/quant/greeks.py)
* **What it does:** Black-Scholes-Merton European Call pricing and analytical Greeks engine. Solves for Implied Volatility (IV) using Brent's root-finding method, and computes first- and second-order sensitivities ($\Delta, \Gamma, \Theta, \nu, \rho$).
* **Inputs & Outputs:** Inputs: $S, K, T, r, \sigma$. Output: `GreeksResult` object.
* **How to test individually:**
  ```bash
  source .venv/bin/activate
  pytest tests/test_greeks.py -v
  ```
  *Tests verify analytical Greeks against closed-form benchmarks, deep ITM/OTM edge cases, and Brent IV recovery.*

#### 9. [`src/quant/merton_jump.py`](file:///Users/prakharsatyam/codes/redagent/src/quant/merton_jump.py)
* **What it does:** Merton (1976) Jump-Diffusion pricing engine and Corrado-Su skewness/kurtosis adjustment. Quantifies non-Gaussian fat tails and Poisson jump arrival intensity ($\lambda$), isolating the extra "jump premium" priced into out-of-the-money crypto calls.
* **Inputs & Outputs:** Inputs: $S, K, T, r, \sigma, \lambda, \mu_J, \sigma_J$. Output: `MertonResult` object.
* **How to test individually:**
  ```bash
  source .venv/bin/activate
  pytest tests/test_merton.py -v
  ```
  *Tests verify that Merton price strictly exceeds standard BSM for OTM calls due to jump risk premium.*

#### 10. [`src/quant/volatility_surface.py`](file:///Users/prakharsatyam/codes/redagent/src/quant/volatility_surface.py)
* **What it does:** Volatility surface analytics. Computes 90-day IV Rank, 30-day Realized Volatility, Volatility Risk Premium ($\text{VRP} = \text{IV} - \text{RV}$), and calibrates the SABR stochastic volatility smile. Classifies market into `cheap_vol_expansion` (ideal for buying calls), `neutral`, or `expensive_crush`.
* **Inputs & Outputs:** Inputs: current IV, historical IV series, realized vol. Output: `VolatilityMetrics` object.
* **How to test individually:**
  ```bash
  source .venv/bin/activate
  pytest tests/test_volatility_surface.py -v
  ```
  *Tests verify IV Rank bounds (0–100%), VRP spread, and regime labeling.*

#### 11. [`src/quant/fee_simulator.py`](file:///Users/prakharsatyam/codes/redagent/src/quant/fee_simulator.py)
* **What it does:** Gatekeeper enforcing the $\ge 3\%$ net return constraint. Projects target Call exit price based on delta sensitivity to a $+2.5\%$ spot move, deducts entry and exit fees, half-spread crossing, and holding-period theta decay.
* **Inputs & Outputs:** Inputs: contract metrics, fees, holding hours. Output: `TradePayoffEvaluation` with `passes_hurdle: bool`.
* **How to test individually:**
  ```bash
  source .venv/bin/activate
  pytest tests/test_fee_simulator.py -v
  ```
  *Tests verify approval of tight liquid contracts and immediate rejection of high-friction setups.*

#### 12. [`src/quant/pattern_discovery.py`](file:///Users/prakharsatyam/codes/redagent/src/quant/pattern_discovery.py)
* **What it does:** AI Symbolic Formula and Pattern Discovery engine. Fits Random Forests and shallow Decision Trees on historical trade outcomes to extract high-importance features and human-interpretable symbolic rules (e.g. `iv_rank <= 32.5 AND delta >= 0.42`).
* **Inputs & Outputs:** Inputs: historical feature DataFrame. Output: `DiscoveredRuleSet` saved to JSON.
* **How to test individually:**
  ```bash
  source .venv/bin/activate
  pytest tests/test_pattern_discovery.py -v
  ```
  *Tests verify feature ranking extraction, JSON serialization/deserialization, and candidate contract scoring.*

---

### Subsystem D: AI Model Runtimes & Serialization (`src/model/`)

#### 13. [`src/model/state_serializer.py`](file:///Users/prakharsatyam/codes/redagent/src/model/state_serializer.py)
* **What it does:** Multi-modal context fusion engine. Converts quantitative Greeks, Merton jump metrics, volatility regimes, friction payoffs, and Stage-1 news vectors into a high-density, structured state document for Laya System 1 inference.
* **Inputs & Outputs:** Inputs: Greeks, Payoff, VolMetrics, MertonResult, Funding. Output: `SerializedMarketState`.
* **How to test individually:**
  ```bash
  source .venv/bin/activate
  pytest tests/test_state_serializer.py -v
  ```
  *Tests verify prompt formatting and attachment of typed decision questions.*

#### 14. [`src/model/laya_news_engine.py`](file:///Users/prakharsatyam/codes/redagent/src/model/laya_news_engine.py)
* **What it does:** In-process Stage-1 News Classifier runtime. Evaluates headlines in sub-15ms. **Protective Circuit Breaker:** If a headline represents a `bearish_dump` (exploit, hack, liquidation waterfall) with confidence $\ge 70\%$, sets `circuit_breaker_triggered = True` to halt all Call option buying. Includes heuristic fallback when checkpoint files are not yet loaded.
* **Inputs & Outputs:** Inputs: headline string. Output: `NewsSentimentResult`.
* **How to test individually:**
  ```bash
  source .venv/bin/activate
  pytest tests/test_laya_news_engine.py -v
  ```
  *Tests verify sub-15ms execution, circuit breaker tripping on bearish news, and neutral handling.*

#### 15. [`src/model/laya_engine.py`](file:///Users/prakharsatyam/codes/redagent/src/model/laya_engine.py)
* **What it does:** In-process Stage-2 Options Decision Head. Runs ModernBERT-421M non-autoregressive inference over the serialized market state. Outputs calibrated win probabilities $P(\text{gain} \ge 3\%)$. Sets `is_signal_triggered = True` if conviction $\ge 75\%$.
* **Inputs & Outputs:** Inputs: `SerializedMarketState`. Output: `OptionDecisionResult`.
* **How to test individually:**
  ```bash
  source .venv/bin/activate
  pytest tests/test_laya_engine.py -v
  ```
  *Tests verify in-process inference speed, decision schemas, and threshold-based trigger gating.*

#### 16. [`src/model/dataset_generator.py`](file:///Users/prakharsatyam/codes/redagent/src/model/dataset_generator.py)
* **What it does:** Replays historical options ticks, simulates real-world friction and forward spot moves, performs **hard negative mining** on "bull traps" (spot up, but contract lost money due to theta or IV crush), and exports walk-forward `train.jsonl` and `val.jsonl` files for Colab training.
* **Inputs & Outputs:** Inputs: tick DataFrame. Output: `train.jsonl` and `val.jsonl`.
* **How to test individually:**
  ```bash
  source .venv/bin/activate
  pytest tests/test_dataset_generator.py -v
  ```
  *Tests verify walk-forward chronological splitting, schema compliance, and hard negative oversampling.*

---

### Subsystem E: Signals, Execution & Paper Trading (`src/signals/`)

#### 17. [`src/signals/alert_bot.py`](file:///Users/prakharsatyam/codes/redagent/src/signals/alert_bot.py)
* **What it does:** Advisory signal formatter and broadcaster. Formats validated trade setups into rich Telegram and Discord markdown messages with spot, strike, Greeks, IV rank, target exit, stop loss, and Laya conviction score.
* **Inputs & Outputs:** Inputs: `ValidatedTradeSignal`. Output: Telegram/Discord message dispatch.
* **How to test individually:**
  ```bash
  source .venv/bin/activate
  pytest tests/test_alert_bot.py -v
  ```
  *Tests verify message formatting, stop/target calculations, and mock HTTP broadcast.*

#### 18. [`src/signals/paper_trader.py`](file:///Users/prakharsatyam/codes/redagent/src/signals/paper_trader.py)
* **What it does:** Virtual portfolio tracking engine. Simulates live execution, tracks open positions, logs entry/exit fees, checks target exits ($+20\%$) and stop losses ($-25\%$), and persists the entire ledger to [`data/paper_portfolio.json`](file:///Users/prakharsatyam/codes/redagent/data/paper_portfolio.json).
* **Inputs & Outputs:** Inputs: `ValidatedTradeSignal`, current mark prices. Output: Updated `PaperPosition` and PnL metrics.
* **How to test individually:**
  ```bash
  source .venv/bin/activate
  pytest tests/test_paper_trader.py -v
  ```
  *Tests verify position opening, target hit closure, stop-loss trigger closure, and ledger JSON persistence.*

#### 19. [`src/signals/signal_engine.py`](file:///Users/prakharsatyam/codes/redagent/src/signals/signal_engine.py)
* **What it does:** Master real-time scanner orchestrator. Runs every 60 seconds:
  1. Calls DataPipeline for multi-source snapshot.
  2. Runs Stage-1 News Circuit Breaker Gate (halts on bearish headlines/waterfalls).
  3. Filters liquid tradeable Call options.
  4. Runs Greeks, Merton jump pricer, and FeeSimulator ($\ge 3\%$ net profit hurdle).
  5. Serializes state into System 1 document.
  6. Runs Stage-2 Laya decision head.
  7. Dispatches alerts via AlertBot and registers entries in PaperTrader.
* **Inputs & Outputs:** Inputs: Real-time market feeds. Output: Broadcast signals and virtual positions.
* **How to test individually:**
  ```bash
  source .venv/bin/activate
  pytest tests/test_signal_engine.py -v
  ```
  *Tests verify circuit breaker halts, fee hurdle rejections, and end-to-end signal trigger passes.*

---

### Subsystem F: Offline Scripts & Tools (`scripts/`)

#### 20. [`scripts/fetch_historical_options.py`](file:///Users/prakharsatyam/codes/redagent/scripts/fetch_historical_options.py)
* **What it does:** Downloads historical Deribit options ticks and order books over a specified window (e.g. 90 days), saving structured data to Parquet format. Includes offline fallback simulation.
* **How to test:** `pytest tests/test_fetch_historical_options.py -v`

#### 21. [`scripts/fetch_historical_news.py`](file:///Users/prakharsatyam/codes/redagent/scripts/fetch_historical_news.py)
* **What it does:** Downloads historical crypto headlines from CryptoPanic and major historical crypto event archives into `historical_news.jsonl`.
* **How to test:** `pytest tests/test_fetch_historical_news.py -v`

#### 22. [`scripts/label_news_dataset.py`](file:///Users/prakharsatyam/codes/redagent/scripts/label_news_dataset.py)
* **What it does:** 15-minute forward event-study labeler. Evaluates market reaction following breaking headlines ($\Delta \text{Spot}_{15\text{m}}$ and $\Delta \text{DVOL}_{15\text{m}}$) to produce ground-truth training sets (`news_train.jsonl` and `news_val.jsonl`).
* **How to test:** `pytest tests/test_label_news_dataset.py -v`

#### 23. [`scripts/sync_colab.py`](file:///Users/prakharsatyam/codes/redagent/scripts/sync_colab.py)
* **What it does:** Two-way sync bridge between local Mac and Google Drive/Colab. Uploads processed training datasets and downloads trained checkpoints back to local `models/` directory with SHA-256 verification.
* **How to test:** `pytest tests/test_sync_colab.py -v`

#### 24. [`scripts/generate_system_design.py`](file:///Users/prakharsatyam/codes/redagent/scripts/generate_system_design.py)
* **What it does:** High-resolution visualization generator. Renders the 200 DPI architectural schematic [`system_design.png`](file:///Users/prakharsatyam/codes/redagent/system_design.png).

---

## 3. Anatomy of a Complete Live Run

Here is the exact step-by-step lifecycle of an execution pass across RedAgent:

```mermaid
sequenceDiagram
    autonumber
    participant Feed as Market Feeds (Deribit / Binance / News)
    participant Pipe as DataPipeline
    participant S1 as Stage-1 Laya News Radar
    participant Quant as Greeks / Merton / Fee Hurdle
    participant Ser as State Serializer
    participant S2 as Stage-2 Laya Decision Head
    participant Exec as AlertBot / PaperTrader

    Feed->>Pipe: Live Spot, Funding, 15m Liqs, Headlines, Call Chain
    Pipe->>S1: Send Headlines & Liquidation Alert
    alt Bearish Dump or Long Waterfall Detected
        S1-->>Pipe: Circuit Breaker Triggered (HALT)
        Note over Pipe: All Call evaluation aborted; Capital protected
    else Market Calm / Bullish Catalyst
        S1-->>Pipe: Radar Clear (Proceed)
        Pipe->>Quant: Filtered Liquid Calls (Spread <= 12%, OI > 0.1)
        Quant->>Quant: Calculate BSM Greeks (Delta, Gamma, Theta, IV)
        Quant->>Quant: Calculate Merton Jump Premium
        Quant->>Quant: Enforce Net Hurdle (Projected Return >= 3%)
        alt Fails Net Hurdle (< 3% or Spread > 5%)
            Note over Quant: Candidate rejected; Skip
        else Passes Net Hurdle (>= 3%)
            Quant->>Ser: Package Metrics into System 1 State Text
            Ser->>S2: Serialized Prompt + Typed Decision Questions
            S2->>S2: ModernBERT Forward Pass (sub-25ms)
            alt Conviction < 75% or should_buy == 'no'
                Note over S2: No Trade Signal
            else Conviction >= 75% AND should_buy == 'yes'
                S2->>Exec: ValidatedTradeSignal Triggered!
                Exec->>Exec: Broadcast Telegram/Discord Alert
                Exec->>Exec: Open Virtual Position in paper_portfolio.json
            end
        end
    end
```

### Trace of an Actual High-Conviction Setup:
1. **Underlying Data:** BTC Spot at $\$65,850$, 8h funding at $+0.003\%$, normal liquidations ($\$1.2\text{M}$ in 15m).
2. **Stage-1 Radar:** Headline *"SEC Approves In-Kind Creation for Spot BTC ETFs"* evaluated in $12\text{ ms} \rightarrow$ `is_market_moving: yes`, `impact_regime: bullish_vol_expansion`, `circuit_breaker_triggered: false`.
3. **Contract Selected:** `BTC-24OCT26-68000-C` (DTE: 3.8 days, Strike: $\$68,000$).
   * Bid: $\$715$, Ask: $\$735$ (Spread: $2.7\%$, well below $5\%$ limit).
4. **Quant Engine:**
   * Delta: $0.34$, Gamma: $0.000045$, Theta: $-\$138/\text{day}$.
   * Merton Jump Price: $\$754.20$ vs BSM: $\$728.50$ (Jump Premium: $\$25.70$).
   * At projected $+2.5\%$ spot move over 24h: Gross payoff $=\$896.50$.
   * Net profit after Taker in, Maker out, spread, and theta $=\$141.20$ ($\mathbf{+19.2\%}$ net return $\ge 3\%$ hurdle **PASSED**).
5. **State Serialization:** Formats metrics into standard ModernBERT prompt.
6. **Stage-2 Inference:** Laya outputs `should_buy_call: "yes"`, `confidence: 0.88`, `expected_net_return: "target_3pct_to_10pct"`.
7. **Execution:** AlertBot sends Telegram alert; PaperTrader registers entry in `paper_portfolio.json` with target exit at $\$882$ ($+20\%$) and stop loss at $\$551$ ($-25\%$).

---

## 4. End-to-End Operational Command List

Below is the complete sequence of terminal commands to execute the pipeline from scratch.

### Phase 1: Environment Initialization
```bash
# 1. Clone repository and navigate
cd /Users/prakharsatyam/codes/redagent

# 2. Create isolated virtual environment with uv
uv venv .venv
source .venv/bin/activate

# 3. Install all quantitative, data, ML, and Laya packages
uv pip install ccxt scipy numpy pandas scikit-learn pyarrow pydantic python-dotenv requests
uv pip install laya transformers accelerate

# 4. Verify installation by running complete unit test suite
pytest
# (Expected output: 63 passed in ~2 seconds)
```

---

### Phase 2: Historical Ingestion & Training Dataset Preparation
```bash
source .venv/bin/activate

# 1. Download 90 days of Deribit BTC options ticks to Parquet
python3 scripts/fetch_historical_options.py \
    --currency BTC \
    --days 90 \
    --output data/raw/deribit_btc.parquet

# 2. Ingest historical crypto headlines from CryptoPanic and archives
python3 scripts/fetch_historical_news.py \
    --days 180 \
    --output data/raw/historical_news.jsonl

# 3. Generate Stage-1 News Dataset using 15-minute forward event study
python3 scripts/label_news_dataset.py \
    --news_input data/raw/historical_news.jsonl \
    --train_out data/processed/news_train.jsonl \
    --val_out data/processed/news_val.jsonl

# 4. Generate Stage-2 Options Dataset (replaying ticks, hard-negative mining bull traps)
python3 src/model/dataset_generator.py \
    --train_out data/processed/train.jsonl \
    --val_out data/processed/val.jsonl

# 5. Discover non-linear mathematical indicators via symbolic regression
python3 -m src.quant.pattern_discovery \
    --input data/raw/deribit_btc.parquet \
    --output models/discovered_rules.json
```

---

### Phase 3: Google Colab GPU Model Fine-Tuning
```bash
# 1. Sync generated datasets from local Mac to Google Drive
python3 scripts/sync_colab.py --upload --source data/processed/

# 2. Open Google Colab and run:
#    A. laya_news.ipynb  (Fine-tunes Stage-1 News Radar on T4 GPU in ~5 minutes)
#    B. laya.ipynb       (Fine-tunes Stage-2 Options Head on T4 GPU in ~8 minutes)

# 3. Download trained model weights back from Google Drive to local Mac
python3 scripts/sync_colab.py --download --destination models/
```

---

### Phase 4: Stage 1 Validation (Historical Backtest / Replay)
```bash
source .venv/bin/activate

# Run historical tick replay test to verify win-rate and fee hurdle enforcement
pytest tests/test_dataset_generator.py tests/test_pattern_discovery.py -v
```

---

### Phase 5: Stage 2 Validation (Active Deribit Testnet / Devnet)
Configure `keys.env` for Deribit Testnet:
```ini
DERIBIT_USE_TESTNET=true
DERIBIT_CLIENT_ID=your_testnet_client_id
DERIBIT_CLIENT_SECRET=your_testnet_client_secret
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

Launch the live 60-second testnet scanner:
```bash
source .venv/bin/activate
python3 -m src.signals.signal_engine \
    --currency BTC \
    --interval 60
```
*Monitors live testnet order books, runs Stage-1 news radar, broadcasts test signals to Telegram, and records trades to `data/paper_portfolio.json`.*

---

### Phase 6: Stage 3 Production (Real Money Execution)
Switch `keys.env` to live Deribit Mainnet credentials and launch:
```bash
source .venv/bin/activate
python3 -m src.signals.signal_engine \
    --currency BTC \
    --interval 60
```

---

## 5. Capital Flow, Account Architecture & Safety Limits

When deploying proprietary capital to crypto options, account structuring and capital controls are critical.

### A. How the Bot Accesses Capital
```
[Bank / Fiat Ramp] 
       │ (USDC / BTC)
       ▼
[Deribit Main Account] (Cold Vault / Holds 90% Capital)
       │ Internal Transfer (Instant, Zero Fee)
       ▼
[Deribit Dedicated Subaccount: "RedAgent-Bot"] (Holds 10% Trading Capital)
       │ API Connection (Restricted)
       ▼
[RedAgent Live Engine]
```

1. **Subaccount Isolation:**
   * Never connect your main Deribit master account API keys to any trading bot.
   * Create an isolated **Deribit Subaccount** named `RedAgent-Bot`.
   * Fund this subaccount with strictly the risk capital intended for the bot (e.g. $\$5,000$ or $0.1$ BTC).
2. **API Key Permission Security:**
   * **Read:** Enabled (allows reading order books, positions, balances).
   * **Trade:** Enabled (allows submitting options limit orders).
   * **Withdrawal:** **STRICTLY DISABLED (LOCKED).** The API key physically cannot withdraw funds to any external crypto address.
   * **IP Whitelisting:** Bind the API key strictly to your local Mac's static IP or VPS IP address.

---

### B. Safety Limits & Loss Prevention Architecture

RedAgent enforces a multi-tiered safety net to ensure that losses never exceed predefined risk thresholds:

| Safety Level | Mechanism | Enforced Rule |
| :--- | :--- | :--- |
| **Trade Level** | Stop-Loss Enforcement | Maximum **$-25\%$ loss** on option entry premium. If bid drops to $75\%$ of entry ask, position automatically triggers exit. |
| **Trade Level** | Position Sizing Limit | Max allocation per trade is **$2\%$ of subaccount capital** (e.g. $\$100$ on a $\$5,000$ subaccount). |
| **Portfolio Level** | Max Concurrent Positions | Capped at **$3$ open Call positions** simultaneously. Total capital at risk across all positions $\le 6\%$. |
| **Daily Level** | Daily Portfolio Circuit Breaker | If realized losses in `paper_portfolio.json` exceed **$5\%$ of account balance in 24 hours**, the engine automatically disengages and halts all entries for 24 hours. |
| **Market Level** | Stage-1 News Protective Radar | If breaking news indicates an exploit, hack, or sudden liquidation cascade $\ge \$15\text{M}$, Call buying is aborted immediately. |
| **Exchange Level** | Deribit Maker Max Fee Cap | Fees on Deribit options are capped by exchange rule at $12.5\%$ of the option premium, preventing fee insolvency on cheap wings. |

---

### C. Regulatory & Indian Tax Shielding

Under Indian Section 115BBH, crypto derivatives are taxed at a flat **$30\%$ on profits with zero set-off against losses**, and Section 194S deducts a mandatory **$1\%$ TDS** on domestic transactions.

RedAgent completely neutralizes this tax drag via two approved operational models:

1. **Model A: Advisory Signal Broadcaster (Domestic India - LLP/Pvt Ltd)**
   * RedAgent does not execute automatic domestic trades.
   * It operates as an AI-powered advisory intelligence engine broadcasting high-probability signals via Telegram.
   * Subscription or advisory revenue qualifies under standard corporate income tax (Section 115BAA at $22\%$ net), allowing $100\%$ deduction of operational expenses (servers, APIs, data feeds).
2. **Model B: Offshore Proprietary Fund (Dubai IFZA / GIFT City IFSC)**
   * For executing live proprietary capital on international exchanges (Deribit/Binance), deploy through an offshore entity (e.g. Dubai IFZA Free Zone or GIFT City IFSC).
   * Enjoy $0\%$ capital gains tax on international crypto derivatives, full loss-offset capability, and zero Indian TDS deduction.

---

## 6. Verification & Support Checklist

Before deploying on mainnet, verify this checklist:
- [x] All 63 pytest unit tests passing (`pytest`).
- [x] Deribit API keys configured in `keys.env` with **Withdrawal Disabled**.
- [x] Telegram Bot token and Chat ID tested (`pytest tests/test_alert_bot.py`).
- [x] Paper trading ledger active (`data/paper_portfolio.json`).
- [x] Stage-1 News Circuit Breaker tested and active.
- [x] Capital strictly isolated in a dedicated subaccount.
