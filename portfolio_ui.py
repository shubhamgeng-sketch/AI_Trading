"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Portfolio Optimizer UI  v3                                                 ║
║  Run:  streamlit run portfolio_ui_v3.py                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, glob, json, math
import pandas as pd
import numpy  as np
import streamlit as st

from temp import PortfolioOptimizer, print_allocation

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Portfolio Optimizer v3",
    page_icon  = "📊",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

st.title("📊 Portfolio Optimizer  —  v3 (Quant)")
st.caption(
    "Powered by Cross-Asset Transformer + Dirichlet Policy  |  "
    "Fixes: uniform-weight collapse, wrong entropy target, no cross-sectional signal"
)
st.markdown("---")

current_dir = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE LOADERS  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models …")
def load_models() -> dict:
    models = {}
    for name, fname in [("PPO", "ppo_final_v3.pt"), ("SAC", "sac_final_v3.pt")]:
        path = os.path.join(current_dir, fname)
        if os.path.exists(path):
            try:
                models[name] = PortfolioOptimizer(path, deterministic=True)
            except Exception as e:
                st.error(f"Failed to load {name} model: {e}")
        else:
            st.warning(f"⚠️  {fname} not found in {current_dir}")
    return models


@st.cache_data(show_spinner="Loading price data …")
def load_all_csv() -> pd.DataFrame:
    frames = {}
    for csv_file in sorted(glob.glob(os.path.join(current_dir, "*_daily.csv"))):
        ticker = os.path.basename(csv_file).replace("_daily.csv", "").upper()
        try:
            df = pd.read_csv(csv_file)
            if "close" not in df.columns:
                continue
            df = df[["timestamp", "close"]].copy()
            df.columns = ["Date", ticker]
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            frames[ticker] = df[ticker]
        except Exception:
            continue
    return pd.DataFrame(frames) if frames else pd.DataFrame()


# ── Load ──────────────────────────────────────────────────────────────────────
models    = load_models()
prices_all = load_all_csv()

if not models:
    st.error("❌ No v3 models found.  Train with `drl_portfolio_v3.py` first.")
    st.stop()
if prices_all.empty:
    st.error("❌ No `*_daily.csv` files found.")
    st.stop()

# Tickers available in BOTH the model universe AND loaded CSV data
first_opt          = next(iter(models.values()))
universe_tickers   = first_opt.available_tickers()
available_tickers  = [t for t in universe_tickers if t in prices_all.columns]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️  Configuration")

    model_choice = st.radio(
        "Model",
        options=["PPO", "SAC", "Both"],
        help=(
            "**PPO**: on-policy, tends toward concentrated bets.\n\n"
            "**SAC**: off-policy Dirichlet, moderate diversification.\n\n"
            "**Both**: run and compare side-by-side."
        ),
    )

    st.divider()
    st.header("📈  Stock Selection")

    # Quick-select sector groups
    SECTORS = {
        "All":     available_tickers,
        "Banking": [t for t in available_tickers if t in
                    ["AXISBANK","HDFCBANK","ICICIBANK","INDUSINDBK","KOTAKBANK",
                     "SBIN","BANKBARODA","PNB","UNIONBANK"]],
        "IT":      [t for t in available_tickers if t in
                    ["TCS","INFY","HCLTECH","WIPRO","MPHASIS","COFORGE"]],
        "FMCG":    [t for t in available_tickers if t in
                    ["HINDUNILVR","ITC","DABUR","MARICO","GODREJCP","COLPAL",
                     "BRITANNIA","TATACONSUM"]],
    }

    sector_preset = st.selectbox("Quick-select sector", list(SECTORS.keys()))
    default_picks = SECTORS[sector_preset][:8]

    selected_stocks = st.multiselect(
        "Choose stocks",
        options  = available_tickers,
        default  = default_picks,
        help     = "Pick 1–30 stocks from the master universe.",
    )

    st.divider()
    st.header("🔧  Inference Options")

    days_to_load = st.slider(
        "History (trading days)",
        min_value = 50,
        max_value = 500,
        value     = 252,
        step      = 10,
        help      = "252 ≈ 1 trading year.  Minimum needed = 40.",
    )

    top_n = st.slider(
        "Top-N holdings (0 = no filter)",
        min_value = 0,
        max_value = 20,
        value     = 0,
        step      = 1,
    )

    min_weight = st.slider(
        "Minimum weight threshold",
        min_value = 0.0,
        max_value = 0.05,
        value     = 0.005,
        step      = 0.001,
        format    = "%.3f",
    )

    st.divider()
    st.success(
        f"✓  {len(models)} model(s) loaded\n\n"
        f"✓  {len(available_tickers)} tickers available"
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PANEL — status row
# ─────────────────────────────────────────────────────────────────────────────
if not selected_stocks:
    st.warning("⚠️  Please select at least one stock in the sidebar to continue.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Selected Stocks", len(selected_stocks))
c2.metric("History Days",    days_to_load)
c3.metric("Model(s)",        model_choice)
c4.metric("Universe Size",   len(available_tickers))

# ── Prepare data ──────────────────────────────────────────────────────────────
prices_sub = prices_all[selected_stocks].dropna()
if len(prices_sub) > days_to_load:
    prices_sub = prices_sub.iloc[-days_to_load:]

st.info(
    f"📅  {prices_sub.shape[0]} trading days × {prices_sub.shape[1]} stocks  |  "
    f"{prices_sub.index[0].date()} → {prices_sub.index[-1].date()}"
)

# ── Price chart (collapsible) ─────────────────────────────────────────────────
with st.expander("📉  Price chart (selected stocks — normalised)"):
    norm_px = prices_sub / prices_sub.iloc[0]
    st.line_chart(norm_px, height=260)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# RUN BUTTON
# ─────────────────────────────────────────────────────────────────────────────
run = st.button(
    "🚀  Generate Portfolio Allocation",
    type             = "primary",
    use_container_width = True,
)

if run:
    models_to_run = (
        ["PPO"]                if model_choice == "PPO"
        else ["SAC"]           if model_choice == "SAC"
        else list(models.keys())
    )

    results: dict = {}
    for algo in models_to_run:
        if algo not in models:
            st.warning(f"Model {algo} not loaded — skipping.")
            continue
        with st.spinner(f"Running {algo} inference …"):
            try:
                results[algo] = models[algo].predict(
                    tickers    = selected_stocks,
                    price_data = prices_sub,
                    top_n      = top_n if top_n > 0 else None,
                    min_weight = min_weight,
                )
            except Exception as e:
                st.error(f"[{algo}] {e}")
                import traceback; traceback.print_exc()

    if not results:
        st.error("No allocations generated."); st.stop()

    # ─────────────────────────────────────────────────────────────────────
    # RESULTS DISPLAY
    # ─────────────────────────────────────────────────────────────────────
    st.header("📊  Results")

    for algo, result in results.items():
        alloc   = result["allocations"]
        summary = result["summary"]

        # Uniform HHI reference for this basket
        uniform_hhi = summary["uniform_hhi"]
        hhi         = summary["hhi"]

        # Concentration badge
        if hhi <= uniform_hhi * 1.05:
            conc_badge   = "🟡 Near-uniform (check model)"
            conc_delta   = "≈ uniform"
        elif hhi < uniform_hhi * 2:
            conc_badge   = "🟢 Moderate concentration (healthy)"
            conc_delta   = f"+{(hhi/uniform_hhi - 1)*100:.0f}% vs uniform"
        else:
            conc_badge   = "🔵 High concentration"
            conc_delta   = f"+{(hhi/uniform_hhi - 1)*100:.0f}% vs uniform"

        with st.container():
            st.subheader(f"{algo} Allocation")
            st.caption(f"Concentration: {conc_badge}")

            # ── KPI row ───────────────────────────────────────────────────
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Active Stocks",  summary["n_stocks"])
            k2.metric("Invested",       f"{summary['invested_pct']:.1f}%")
            k3.metric("Cash",           f"{summary['cash_pct']:.1f}%")
            k4.metric("Top Holding",    summary["top_holding"])
            k5.metric("HHI",            f"{hhi:.4f}", delta=conc_delta,
                      help=f"Uniform HHI for {len(selected_stocks)} stocks = {uniform_hhi:.4f}. "
                           "Higher → more selective.")
            k6.metric("Top-5 Share",    f"{summary['top5_share']:.1f}%")

            # ── Allocation table + bar chart side-by-side ─────────────────
            tbl_col, chart_col = st.columns([1, 1])

            with tbl_col:
                st.caption("Allocation Table")
                rows = [
                    {"Stock": k, "Weight (%)": v}
                    for k, v in alloc.items() if v > 0
                ]
                df_alloc = pd.DataFrame(rows)
                st.dataframe(df_alloc, hide_index=True, use_container_width=True,
                             height=min(38 + 35 * len(df_alloc), 450))

            with chart_col:
                st.caption("Allocation Bar Chart")
                chart_data = {
                    k: v for k, v in alloc.items()
                    if v > 0 and k != "CASH"
                }
                if chart_data:
                    chart_df = (
                        pd.DataFrame.from_dict(
                            chart_data, orient="index", columns=["Weight (%)"])
                        .sort_values("Weight (%)", ascending=False)
                    )
                    st.bar_chart(chart_df, use_container_width=True, height=280)

            # ── Concentration visualiser ──────────────────────────────────
            st.caption("🔎  HHI gauge vs equal-weight baseline")
            prog_val = min(hhi / max(1.0 / max(len(selected_stocks) * 0.3, 1), 1e-9), 1.0)
            st.progress(
                float(np.clip(prog_val, 0.0, 1.0)),
                text=(
                    f"HHI = {hhi:.4f}  |  "
                    f"Uniform 1/N = {uniform_hhi:.4f}  |  "
                    f"Ratio = {hhi / uniform_hhi:.2f}×  "
                    f"({'selective ✓' if hhi > uniform_hhi * 1.1 else 'near-uniform ⚠'})"
                )
            )

            # ── JSON export ───────────────────────────────────────────────
            with st.expander("💾  Export as JSON"):
                st.json(alloc)
                st.download_button(
                    label     = f"Download {algo} allocation.json",
                    data      = json.dumps(alloc, indent=2),
                    file_name = f"{algo.lower()}_allocation.json",
                    mime      = "application/json",
                )

        st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────
    # SIDE-BY-SIDE COMPARISON (if both run)
    # ─────────────────────────────────────────────────────────────────────
    if len(results) > 1:
        st.header("🔍  Model Comparison")

        comp_rows = []
        for algo, result in results.items():
            s = result["summary"]
            comp_rows.append({
                "Model"          : algo,
                "Active Stocks"  : s["n_stocks"],
                "Top Holding"    : s["top_holding"],
                "Top Holding %"  : f"{result['allocations'].get(s['top_holding'], 0):.2f}%",
                "Top-5 Share %"  : f"{s['top5_share']:.1f}%",
                "HHI"            : f"{s['hhi']:.4f}",
                "Uniform HHI"    : f"{s['uniform_hhi']:.4f}",
                "HHI Ratio"      : f"{s['hhi'] / s['uniform_hhi']:.2f}×",
                "Cash %"         : f"{s['cash_pct']:.2f}%",
            })

        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

        # ── Overlap bar — how many stocks both models agree on ────────────
        allocs = {a: set(k for k, v in r["allocations"].items()
                         if k != "CASH" and v > 0)
                  for a, r in results.items()}
        algo_names = list(allocs.keys())
        if len(algo_names) == 2:
            a1, a2    = algo_names
            overlap   = allocs[a1] & allocs[a2]
            only_a1   = allocs[a1] - allocs[a2]
            only_a2   = allocs[a2] - allocs[a1]
            ov1, ov2, ov3 = st.columns(3)
            ov1.metric(f"Only {a1}",    len(only_a1),
                       help=", ".join(sorted(only_a1)) or "—")
            ov2.metric("Both models",   len(overlap),
                       help=", ".join(sorted(overlap)) or "—")
            ov3.metric(f"Only {a2}",    len(only_a2),
                       help=", ".join(sorted(only_a2)) or "—")

        st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#888; font-size:12px; padding:18px 0 6px;'>
    Portfolio Optimizer v3  |  Cross-Asset Transformer + Dirichlet SAC  |  NSE Master Universe
</div>
""", unsafe_allow_html=True)