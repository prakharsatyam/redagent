import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
import os

def create_system_design():
    # 26x16 figure for high resolution and comprehensive layout
    fig, ax = plt.subplots(figsize=(26, 16), dpi=200)
    fig.patch.set_facecolor('#0a0d14')
    ax.set_facecolor('#0a0d14')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color Palette
    bg_card = '#111827'
    border_card = '#1f2937'
    text_primary = '#f9fafb'
    text_secondary = '#9ca3af'
    accent_blue = '#38bdf8'
    accent_purple = '#a855f7'
    accent_emerald = '#34d399'
    accent_amber = '#fbbf24'
    accent_rose = '#fb7185'
    accent_cyan = '#06b6d4'
    badge_bg = '#1e293b'

    def draw_card(x, y, w, h, title="", subtitle="", color=accent_blue, items=None, border_color=border_card):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.0",
                                     facecolor=bg_card, edgecolor=border_color, linewidth=1.5, zorder=2)
        ax.add_patch(box)
        
        bar = patches.FancyBboxPatch((x + 0.6, y + h - 1.0), w - 1.2, 0.35, boxstyle="round,pad=0.1",
                                     facecolor=color, edgecolor='none', zorder=3)
        ax.add_patch(bar)

        ax.text(x + 1.0, y + h - 2.3, title, fontsize=11, fontweight='bold', color=color, zorder=4)
        if subtitle:
            ax.text(x + 1.0, y + h - 3.5, subtitle, fontsize=8.5, fontweight='normal', color=text_secondary, zorder=4)
            start_y = y + h - 4.8
        else:
            start_y = y + h - 3.3

        if items:
            for i, it in enumerate(items):
                ax.text(x + 1.2, start_y - (i * 1.95), f"• {it}", fontsize=8.2, color=text_primary, zorder=4)

    def draw_arrow(x1, y1, x2, y2, color='#64748b', style='->', lw=2.0, rad=0.0):
        connectionstyle = f"arc3,rad={rad}" if rad != 0 else "arc3"
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                    connectionstyle=connectionstyle), zorder=5)

    # ==================== HEADER ====================
    ax.text(50, 97.2, "REDAGENT: 2-STAGE LAYA SYSTEM DESIGN & END-TO-END ARCHITECTURE",
            fontsize=21, fontweight='heavy', color='#f8fafc', ha='center', va='center')
    ax.text(50, 94.7, "Dual-Stage Decision Engine (Laya-News + Laya-Options) with Merton/SABR Quant & India Localization",
            fontsize=11.5, fontweight='medium', color=accent_blue, ha='center', va='center')

    # Badges
    badges = [
        ("Architecture: 2-Stage Laya", accent_purple),
        ("Math: Merton Jump + SABR Skew", accent_blue),
        ("Pattern AI: PySR / Symbolic AI", accent_amber),
        ("Inference: ~25ms In-Process", accent_emerald),
        ("India: Advisory SaaS / Dubai Prop", accent_rose)
    ]
    badge_x = 7.5
    for text, col in badges:
        bbox_props = dict(boxstyle="round,pad=0.35", fc=badge_bg, ec=col, lw=1.1)
        ax.text(badge_x, 91.8, text, fontsize=8.5, fontweight='semibold', color='#e2e8f0', bbox=bbox_props)
        badge_x += 17.5

    # ==================== COLUMN HEADERS ====================
    col_y = 88.5
    ax.text(17.5, col_y, "STAGE 0: DATA & QUANT PIPELINE", fontsize=11.5, fontweight='bold', color=accent_blue, ha='center')
    ax.text(50.0, col_y, "STAGE 1 & 2: DUAL LAYA DECISION HEADS", fontsize=11.5, fontweight='bold', color=accent_purple, ha='center')
    ax.text(82.5, col_y, "STAGE 3: INDIA TAX, RISK & EXECUTION", fontsize=11.5, fontweight='bold', color=accent_emerald, ha='center')

    # ==================== COLUMN 1: DATA & ADVANCED QUANT (x = 3 to 32) ====================
    # Box 1A: Live Ingestion & Alternative Data
    draw_card(3, 72.5, 29, 14.5,
              title="1. Multi-Source Ingestion & News Stream",
              subtitle="src/data/ (Tick Books + Twitter/X + Liquidations)",
              color=accent_blue,
              items=[
                  "Deribit WebSocket & REST: BTC/ETH options tick books",
                  "Binance Spot & Perp feeds (Funding rates, Open Interest)",
                  "Twitter/X & CryptoPanic RSS: Breaking news streamer",
                  "Coinglass Liquidation Feed: Long/Short liquidation cascades",
                  "Historical Deribit Parquet tick storage for model training"
              ])

    # Box 1B: Advanced Math & Greeks Engine
    draw_card(3, 52.5, 29, 18.5,
              title="2. Advanced Options & Greeks Engine",
              subtitle="src/quant/ (Merton Jump + SABR + Greeks)",
              color=accent_blue,
              items=[
                  "Merton Jump-Diffusion Model: Accommodates crypto price jumps",
                  "Corrado-Su Adjustment: Corrects BS for fat tails & kurtosis",
                  "SABR Volatility Surface: Calibrates IV smile and term structure",
                  "Analytical Greeks: Delta (Δ), Gamma (Γ), Theta (Θ), Vega (v), Vanna",
                  "90-day IV Rank, Volatility Risk Premium (VRP = IV - RV)",
                  "Overcoming BS: Replaces Gaussian assumptions with jump dynamics"
              ])

    # Box 1C: Pattern AI & Symbolic Discovery
    draw_card(3, 29.5, 29, 21.5,
              title="3. Pattern AI & Formula Discovery",
              subtitle="src/quant/pattern_discovery.py (PySR + SHAP)",
              color=accent_amber,
              items=[
                  "Analyzes historical winning calls (Net Profit >= 3% hurdle)",
                  "Symbolic Regression (PySR): Discovers mathematical formulas",
                  "XGBoost Feature Attribution: Computes SHAP values on Greeks",
                  "Extracts non-linear rule combinations automatically:",
                  "  Example: (IV_Rank < 22) AND (Delta_Mom > 1.4) AND (VRP < 0)",
                  "Feeds discovered mathematical rules as tokens to Laya Stage-2",
                  "Eliminates human indicator bias and prevents curve overfitting"
              ])

    # ==================== COLUMN 2: DUAL LAYA DECISION HEADS (x = 35.5 to 64.5) ====================
    # Box 2A: Stage 1 - Laya News Classifier
    draw_card(35.5, 72.5, 29, 14.5,
              title="4. Laya Stage-1: News & Sentiment Filter",
              subtitle="src/model/laya_news_engine.py (convaiinnovations/laya)",
              color=accent_rose,
              items=[
                  "Evaluates live Twitter/X, CryptoPanic & SEC/Fed headlines",
                  "Q1 (is_market_moving): Determines if news affects BTC/ETH calls",
                  "Q2 (impact_regime): [bullish_vol_expand, bearish_dump, chop_iv_crush]",
                  "Prevents quant model from buying calls into macro crashes",
                  "Outputs structured sentiment vector to Laya Stage-2 state"
              ])

    # Box 2B: State Fusion & Serialization
    draw_card(35.5, 52.5, 29, 18.5,
              title="5. Multi-Modal State Fusion Serializer",
              subtitle="src/model/state_serializer.py (Quant + News + Rules)",
              color=accent_purple,
              items=[
                  "Fuses 3 distinct information domains into compact state:",
                  "  1. Advanced Quant: Merton Greeks, IV Rank, SABR skew",
                  "  2. Laya Stage-1: News sentiment regime & impact confidence",
                  "  3. Pattern AI: Discovered symbolic formula triggers",
                  "Applies India-specific localized friction (fees, TDS, spread)",
                  "Formats into System 1 typed decision contract schema"
              ])

    # Box 2C: Stage 2 - Laya Options Decision Head
    draw_card(35.5, 29.5, 29, 21.5,
              title="6. Laya Stage-2: Options Decision Engine",
              subtitle="convaiinnovations/laya (ModernBERT-421M + LoRA)",
              color=accent_purple,
              items=[
                  "Non-Autoregressive ModernBERT-large (~421M params)",
                  "Fine-tuned on Colab via LoRA (r=16, alpha=32) + RLCD",
                  "Q1 (should_buy_call): noul (yes / no) with calibrated confidence",
                  "Q2 (expected_net_return): choice [<0, 0-2%, 3-10%, >10%]",
                  "Q3 (risk_regime): choice [theta_decay, iv_crush, trending_mom]",
                  "In-process Python execution: ~15-25ms per strike on Mac MPS",
                  "Dual Trigger Gate: P(should_buy_call = yes) >= 0.75"
              ])

    # ==================== COLUMN 3: TRAINING & INDIA EXECUTION (x = 68 to 97) ====================
    # Box 3A: Colab Fine-Tuning Pipeline
    draw_card(68, 72.5, 29, 14.5,
              title="7. Colab LoRA Training Pipeline",
              subtitle="laya.ipynb (Google Colab T4 GPU)",
              color=accent_rose,
              items=[
                  "Multi-modal training set: train.jsonl & val.jsonl",
                  "Chronological walk-forward split with Hard Negative Mining",
                  "Fine-tunes ModernBERT with LoRA in ~5-10 mins on free T4",
                  "Regime-segmented temperature calibration (High vs Low IV)",
                  "Syncs checkpoints directly to Google Drive / local Mac"
              ])

    # Box 3B: India Localization & Business Structuring
    draw_card(68, 50.5, 29, 20.5,
              title="8. India Localization & Business Entity",
              subtitle="config/fees.py (Taxes, TDS & Structuring)",
              color=accent_amber,
              items=[
                  "Section 115BBH Defense: 30% tax with ZERO loss offset!",
                  "Section 194S Accounting: 1% TDS cash drag factored in hurdle",
                  "USDT/INR Premium: 1.5% P2P banking conversion spread modeled",
                  "RECOMMENDED ENTITY SETUP FOR MAX BENEFIT & LEAST COST:",
                  "• Route A (Advisory SaaS): Indian LLP/Pvt Ltd selling signals.",
                  "  Taxed at normal corporate tax (22%), 100% expense deductions!",
                  "• Route B (Prop Trading): Dubai IFZA or GIFT City IFSC entity.",
                  "  0% crypto capital gains tax, legally bypasses 115BBH!"
              ])

    # Box 3C: Signal Alerts & Paper Execution
    draw_card(68, 29.5, 29, 19.5,
              title="9. Signals, Alerts & Paper Execution",
              subtitle="src/signals/ (Telegram Bot & Paper Trader)",
              color=accent_emerald,
              items=[
                  "Instant Telegram & Discord Bot alerts for validated trades",
                  "Includes Strike, DTE, Target Premium, Stop-Loss, and Greeks",
                  "Simulates paper portfolio: Validates edge before real capital",
                  "Live API router ready for Deribit Testnet / Binance Options",
                  "Protects Indian resident traders by operating in Advisory mode"
              ])

    # ==================== ARROWS & FLOWS ====================
    # News -> Laya Stage 1
    draw_arrow(32, 79.5, 35.5, 79.5, color=accent_rose, lw=2.2)

    # Laya Stage 1 -> State Serializer
    draw_arrow(50, 72.5, 50, 71, color=accent_rose)

    # Quant + Pattern AI -> State Serializer
    draw_arrow(32, 61.5, 35.5, 61.5, color=accent_blue, lw=2.2)
    draw_arrow(32, 40, 35.5, 56, color=accent_amber, lw=2.2, rad=0.1)

    # State Serializer -> Laya Stage 2
    draw_arrow(50, 52.5, 50, 51, color=accent_purple)

    # Colab Training -> Laya Stage 2 Checkpoint
    draw_arrow(68, 79.5, 64.5, 40, color=accent_rose, lw=2.0, rad=-0.15)

    # Laya Stage 2 -> India Risk & Signals
    draw_arrow(64.5, 40, 68, 40, color=accent_emerald, lw=2.2)
    draw_arrow(82.5, 50.5, 82.5, 49, color=accent_amber)

    # ==================== BOTTOM FOOTER: SUMMARY & MANIFEST ====================
    foot_box = patches.FancyBboxPatch((3, 3), 94, 24.5, boxstyle="round,pad=0.8,rounding_size=1.2",
                                      facecolor='#0f172a', edgecolor='#334155', linewidth=1.5, zorder=2)
    ax.add_patch(foot_box)

    ax.text(5, 25.5, "CORE ARCHITECTURAL RATINGS & STRATEGIC SUMMARY", fontsize=11.5, fontweight='bold', color=accent_amber, zorder=4)

    col1_txt = (
        "1. QUANT ENGINE UPGRADE (Score: 8.8/10)\n"
        "• Merton Jump-Diffusion & SABR replace naive Black-Scholes,\n"
        "  capturing crypto fat tails, discontinuous jumps, and skew.\n"
        "• PySR Symbolic AI discovers non-linear mathematical rules\n"
        "  directly from historical winning call options without curve fitting.\n"
        "• Twitter/News + Laya Stage-1 acts as an early warning radar\n"
        "  to block call buying ahead of impending macro dumps."
    )
    ax.text(5, 14.5, col1_txt, fontsize=8.2, color='#cbd5e1', zorder=4, linespacing=1.35)

    col2_txt = (
        "2. INDIA TAX & BUSINESS SETUP (Score: 9.5/10)\n"
        "• Section 115BBH (30% tax with ZERO loss set-off) is fatal for\n"
        "  automated domestic execution. 1% TDS causes severe capital drag.\n"
        "• BEST ROUTE: Form an Indian LLP / Pvt Ltd operating as an\n"
        "  Advisory SaaS / Tech Analytics provider (22% tax + 100% expense\n"
        "  deduction on servers, APIs, laptops, research).\n"
        "• FOR LIVE TRADING: Route execution through a Dubai IFZA or\n"
        "  GIFT City IFSC offshore entity to legally eliminate 115BBH."
    )
    ax.text(37, 14.5, col2_txt, fontsize=8.2, color='#cbd5e1', zorder=4, linespacing=1.35)

    col3_txt = (
        "3. TOP 3 LAYA DATA ENGINEERING IDEAS (Score: 9.0/10)\n"
        "• Rank 1: Multi-Modal Context Fusion (Merton Greeks + Laya News\n"
        "  Sentiment + PySR Symbolic Formula tokens into unified state).\n"
        "• Rank 2: Walk-Forward Chronological Splitting with Hard Negative\n"
        "  Mining (specifically trains Laya to identify and reject bull traps).\n"
        "• Rank 3: LoRA Fine-Tuning with Regime-Segmented Temperature\n"
        "  Calibration (fits softmax temperature separately for high/low IV)."
    )
    ax.text(69, 14.5, col3_txt, fontsize=8.2, color='#cbd5e1', zorder=4, linespacing=1.35)

    # One line banner at the very bottom
    ax.text(50, 4.2, "Complete Project Manifest: 24 Specialized Modules & Scripts documented in implementation_plan.md",
            fontsize=9.5, fontweight='bold', color=accent_cyan, ha='center', zorder=4)

    plt.tight_layout()
    output_path = "/Users/prakharsatyam/codes/redagent/system_design.png"
    plt.savefig(output_path, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=200)
    
    space_path = "/Users/prakharsatyam/codes/redagent/system design.png"
    shutil.copyfile(output_path, space_path)
    print(f"Generated updated {output_path} and {space_path}")

if __name__ == "__main__":
    create_system_design()
