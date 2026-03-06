"""
Monte Carlo Option Pricing — Streamlit Dashboard
=================================================
Interactive pricing, Greeks and convergence analysis.
Run:  streamlit run app.py
"""

from __future__ import annotations

import time
import warnings
from datetime import date, timedelta

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.instruments.market import Market
from src.instruments.option_trade import OptionTrade
from src.pricing.black_scholes import BlackScholes
from src.pricing.greeks import AllGreeks, GreeksConfig, MCGreeks
from src.pricing.monte_carlo_model import MonteCarloModel
from src.pricing.regression import BasisType

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MC Option Pricer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS — dark theme  (navy + red + white)
# ─────────────────────────────────────────────────────────────────────────────

NAVY = "#0A1628"
NAVY2 = "#0D1F3C"
RED = "#C8102E"
RED_LIGHT = "#E8374F"
GOLD = "#F5A623"
WHITE = "#F0F4FA"
GREY = "#8B9AB3"
GREEN = "#00C875"

st.markdown(
    f"""
    <style>
    /* ── Base ── */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {NAVY};
        color: {WHITE};
        font-family: 'Segoe UI', sans-serif;
    }}
    [data-testid="stSidebar"] {{
        background-color: {NAVY2};
        border-right: 1px solid #1E3A5F;
    }}
    [data-testid="stSidebar"] * {{
        color: {WHITE} !important;
    }}

    /* ── Headers ── */
    h1 {{ color: {WHITE}; letter-spacing: 1px; }}
    h2 {{ color: {RED_LIGHT}; }}
    h3 {{ color: {GOLD}; }}

    /* ── Metric cards ── */
    [data-testid="metric-container"] {{
        background-color: {NAVY2};
        border: 1px solid #1E3A5F;
        border-left: 4px solid {RED};
        border-radius: 8px;
        padding: 12px 16px;
    }}
    [data-testid="metric-container"] label {{
        color: {GREY} !important;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: {WHITE} !important;
        font-size: 1.8rem;
        font-weight: 700;
    }}
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {{
        font-size: 0.8rem;
    }}

    /* ── Tabs ── */
    [data-testid="stTabs"] [role="tablist"] {{
        border-bottom: 2px solid #1E3A5F;
        gap: 4px;
    }}
    [data-testid="stTabs"] button[role="tab"] {{
        background-color: {NAVY2};
        color: {GREY};
        border-radius: 6px 6px 0 0;
        border: 1px solid #1E3A5F;
        border-bottom: none;
        padding: 8px 20px;
        font-weight: 600;
        transition: all 0.2s;
    }}
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
        background-color: {RED};
        color: {WHITE};
        border-color: {RED};
    }}
    [data-testid="stTabs"] button[role="tab"]:hover {{
        color: {WHITE};
        background-color: #1E3A5F;
    }}

    /* ── Buttons ── */
    [data-testid="stButton"] > button {{
        background-color: {RED};
        color: {WHITE};
        border: none;
        border-radius: 6px;
        font-weight: 700;
        padding: 10px 28px;
        letter-spacing: 0.5px;
        transition: background-color 0.2s;
    }}
    [data-testid="stButton"] > button:hover {{
        background-color: {RED_LIGHT};
    }}

    /* ── Inputs ── */
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] div,
    [data-testid="stSlider"] > div {{
        background-color: #112240 !important;
        color: {WHITE} !important;
        border-color: #1E3A5F !important;
    }}

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {{
        background-color: {NAVY2};
        border: 1px solid #1E3A5F;
        border-radius: 8px;
    }}

    /* ── Divider ── */
    hr {{ border-color: #1E3A5F; }}

    /* ── Info / Warning boxes ── */
    [data-testid="stInfo"] {{
        background-color: #0D2545;
        border-left: 4px solid #1E6FBF;
        color: {WHITE};
    }}
    [data-testid="stWarning"] {{
        background-color: #2A1500;
        border-left: 4px solid {GOLD};
    }}
    [data-testid="stSuccess"] {{
        background-color: #002A1A;
        border-left: 4px solid {GREEN};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib dark style helper
# ─────────────────────────────────────────────────────────────────────────────

def _apply_dark_style(fig: plt.Figure):
    """Apply consistent dark theme to a matplotlib figure."""
    fig.patch.set_facecolor(NAVY)
    for a in fig.get_axes():
        a.set_facecolor(NAVY2)
        a.tick_params(colors=GREY)
        a.xaxis.label.set_color(WHITE)
        a.yaxis.label.set_color(WHITE)
        a.title.set_color(WHITE)
        for spine in a.spines.values():
            spine.set_edgecolor("#1E3A5F")
        a.legend(
            facecolor=NAVY2,
            edgecolor="#1E3A5F",
            labelcolor=WHITE,
        ) if a.get_legend() else None
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown(
        f"<h1 style='margin-bottom:0'>📈 Monte Carlo Option Pricer</h1>"
        f"<p style='color:{GREY}; margin-top:4px; font-size:0.9rem;'>"
        "Black-Scholes · Monte Carlo · Longstaff-Schwartz · Greeks · Convergence"
        "</p>",
        unsafe_allow_html=True,
    )
with col_badge:
    st.markdown(
        f"<div style='background:{RED};border-radius:6px;padding:8px 12px;"
        f"text-align:center;font-size:0.75rem;font-weight:700;margin-top:16px'>"
        f"DAUPHINE<br>2025–2026</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — market & option parameters
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f"<h2 style='color:{RED_LIGHT};font-size:1.1rem;margin-bottom:16px'>"
        "⚙️ Parameters</h2>",
        unsafe_allow_html=True,
    )

    st.markdown(f"<h3 style='color:{GOLD};font-size:0.9rem'>Market</h3>", unsafe_allow_html=True)
    S0 = st.slider("Spot price  S₀", 50.0, 300.0, 100.0, 1.0)
    sigma = st.slider("Volatility  σ", 0.05, 0.80, 0.20, 0.01, format="%.2f")
    r = st.slider("Risk-free rate  r", 0.00, 0.15, 0.05, 0.005, format="%.3f")
    div_a = st.slider("Dividend  D", 0.0, 10.0, 0.0, 0.5)

    st.markdown("---")
    st.markdown(f"<h3 style='color:{GOLD};font-size:0.9rem'>Option</h3>", unsafe_allow_html=True)
    K = st.slider("Strike  K", 50.0, 300.0, 100.0, 1.0)
    T_months = st.slider("Maturity (months)", 1, 36, 12, 1)

    cp = st.selectbox("Call / Put", ["CALL", "PUT"])
    ex = st.selectbox("Exercise", ["EUROPEAN", "AMERICAN"])

    st.markdown("---")
    st.markdown(f"<h3 style='color:{GOLD};font-size:0.9rem'>Monte Carlo</h3>", unsafe_allow_html=True)
    num_paths = st.select_slider(
        "Paths  N",
        options=[1_000, 5_000, 10_000, 50_000, 100_000, 200_000],
        value=50_000,
    )
    antithetic = st.toggle("Antithetic variates", value=True)
    seed = st.number_input("Seed (0 = random)", min_value=0, max_value=9999, value=42)
    seed_val = int(seed) if seed > 0 else None

    if ex == "AMERICAN":
        num_steps = st.slider("Steps (LS)", 50, 252, 100, 10)
    else:
        num_steps = 100

# ─────────────────────────────────────────────────────────────────────────────
# Build Market / OptionTrade objects
# ─────────────────────────────────────────────────────────────────────────────

pricing_date = date.today()
mat_date = pricing_date + timedelta(days=int(T_months * 365 / 12))
ex_div_date = (
    pricing_date + timedelta(days=int(T_months * 365 / 24))
    if div_a > 0
    else pricing_date + timedelta(days=1)
)

market = Market(
    underlying=S0,
    vol=sigma,
    rate=r,
    div_a=div_a,
    ex_div_date=ex_div_date,
)
option = OptionTrade(mat=mat_date, call_put=cp, ex=ex, k=K)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: compute everything
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def compute_prices(
    S0, K, sigma, r, div_a, T_months,
    cp, ex, num_paths, antithetic, seed_val, num_steps,
):
    pricing_date = date.today()
    mat_date = pricing_date + timedelta(days=int(T_months * 365 / 12))
    ex_div_date = (
        pricing_date + timedelta(days=int(T_months * 365 / 24))
        if div_a > 0
        else pricing_date + timedelta(days=1)
    )
    market = Market(S0, sigma, r, div_a, ex_div_date)
    option = OptionTrade(mat=mat_date, call_put=cp, ex=ex, k=K)

    results = {}

    # Black-Scholes (European only)
    if ex == "EUROPEAN":
        t0 = time.perf_counter()
        bs = BlackScholes(market, option, pricing_date)
        bs_price = bs.price()
        results["BS"] = {"price": bs_price, "se": 0.0, "elapsed": time.perf_counter() - t0}
    else:
        results["BS"] = None

    # Monte Carlo
    mc = MonteCarloModel(num_paths, market, option, pricing_date, seed=seed_val)
    t0 = time.perf_counter()
    if ex == "EUROPEAN":
        res = mc.price_european_vectorized(antithetic=antithetic)
    else:
        res = mc.price_american_longstaff_schwartz_vectorized(
            num_steps=num_steps, antithetic=antithetic
        )
    elapsed_mc = time.perf_counter() - t0
    results["MC"] = {
        "price": res["price"],
        "se": res.get("std_error", res.get("se", 0.0)),
        "elapsed": elapsed_mc,
    }
    return results


@st.cache_data(ttl=120)
def compute_greeks(
    S0, K, sigma, r, div_a, T_months, cp, ex, num_paths, antithetic, seed_val,
):
    pricing_date = date.today()
    mat_date = pricing_date + timedelta(days=int(T_months * 365 / 12))
    ex_div_date = (
        pricing_date + timedelta(days=int(T_months * 365 / 24))
        if div_a > 0
        else pricing_date + timedelta(days=1)
    )
    market = Market(S0, sigma, r, div_a, ex_div_date)
    option = OptionTrade(mat=mat_date, call_put=cp, ex=ex, k=K)
    config = GreeksConfig(
        num_paths=min(num_paths, 50_000),
        antithetic=antithetic,
        seed=seed_val,
    )
    g = MCGreeks(market, option, pricing_date, config)
    return g.all_greeks()


@st.cache_data(ttl=120)
def compute_convergence(
    S0, K, sigma, r, div_a, T_months, cp, ex, antithetic, seed_val, num_steps,
):
    pricing_date = date.today()
    mat_date = pricing_date + timedelta(days=int(T_months * 365 / 12))
    ex_div_date = (
        pricing_date + timedelta(days=int(T_months * 365 / 24))
        if div_a > 0
        else pricing_date + timedelta(days=1)
    )
    market = Market(S0, sigma, r, div_a, ex_div_date)
    option = OptionTrade(mat=mat_date, call_put=cp, ex=ex, k=K)
    bs_ref = None
    if ex == "EUROPEAN":
        bs_ref = BlackScholes(market, option, pricing_date).price()

    path_counts = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    rows = []
    for n in path_counts:
        mc = MonteCarloModel(n, market, option, pricing_date, seed=seed_val)
        if ex == "EUROPEAN":
            res = mc.price_european_vectorized(antithetic=antithetic)
        else:
            res = mc.price_american_longstaff_schwartz_vectorized(
                num_steps=num_steps, antithetic=antithetic
            )
        se = res.get("std_error", res.get("se", 0.0))
        rows.append({"N": n, "price": res["price"], "se": se})
    return pd.DataFrame(rows), bs_ref


@st.cache_data(ttl=120)
def compute_smile(
    S0, T_months, r, div_a, ex, num_paths, antithetic, seed_val, num_steps, cp,
):
    """Implied volatility smile (vary strike only)."""
    pricing_date = date.today()
    mat_date = pricing_date + timedelta(days=int(T_months * 365 / 12))
    ex_div_date = (
        pricing_date + timedelta(days=int(T_months * 365 / 24))
        if div_a > 0
        else pricing_date + timedelta(days=1)
    )
    strikes = np.linspace(S0 * 0.70, S0 * 1.30, 13)
    bs_prices, mc_prices = [], []
    for k in strikes:
        mkt = Market(S0, 0.20, r, div_a, ex_div_date)
        opt = OptionTrade(mat=mat_date, call_put=cp, ex=ex, k=float(k))
        bs_prices.append(BlackScholes(mkt, opt, pricing_date).price() if ex == "EUROPEAN" else None)
        mc = MonteCarloModel(num_paths, mkt, opt, pricing_date, seed=seed_val)
        if ex == "EUROPEAN":
            res = mc.price_european_vectorized(antithetic=antithetic)
        else:
            res = mc.price_american_longstaff_schwartz_vectorized(
                num_steps=num_steps, antithetic=antithetic
            )
        mc_prices.append(res["price"])
    return strikes, bs_prices, mc_prices


@st.cache_data(ttl=60)
def simulate_paths(S0, sigma, r, T_months, n_display=50, seed_val=42):
    """Simulate GBM paths for visualization."""
    np.random.seed(seed_val or 0)
    T = T_months / 12
    N = 252
    dt = T / N
    t = np.linspace(0, T, N + 1)
    paths = S0 * np.exp(
        np.cumsum(
            (r - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * np.random.randn(n_display, N),
            axis=1,
        )
    )
    paths = np.hstack([np.full((n_display, 1), S0), paths])
    return t, paths


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_price, tab_paths, tab_greeks, tab_conv, tab_smile, tab_about = st.tabs(
    ["💰 Pricing", "📉 Paths", "🔢 Greeks", "📊 Convergence", "🌊 Strike Profile", "ℹ️ About"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRICING
# ══════════════════════════════════════════════════════════════════════════════

with tab_price:
    st.markdown(f"<h2>Option Pricing</h2>", unsafe_allow_html=True)

    run = st.button("▶  Run Pricing", use_container_width=False)

    if run or "prices_computed" in st.session_state:
        st.session_state["prices_computed"] = True
        with st.spinner("Computing prices…"):
            results = compute_prices(
                S0, K, sigma, r, div_a, T_months,
                cp, ex, num_paths, antithetic, seed_val, num_steps,
            )

        mc_res = results["MC"]
        bs_res = results["BS"]
        mc_price = mc_res["price"]
        mc_se = mc_res["se"]
        mc_ci_lo = mc_price - 1.96 * mc_se
        mc_ci_hi = mc_price + 1.96 * mc_se

        # ── Metrics row ──
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("MC Price", f"{mc_price:.4f}")
        with c2:
            st.metric("MC Std Error", f"{mc_se:.4f}")
        with c3:
            st.metric("95% CI", f"[{mc_ci_lo:.3f}, {mc_ci_hi:.3f}]")
        with c4:
            if bs_res:
                delta = mc_price - bs_res["price"]
                st.metric("BS Price", f"{bs_res['price']:.4f}", delta=f"{delta:+.4f}")
            else:
                st.metric("BS Price", "N/A (American)")

        st.markdown("---")

        # ── Summary table ──
        st.markdown(f"<h3>Results Summary</h3>", unsafe_allow_html=True)
        rows = []
        rows.append({
            "Method": f"Monte Carlo ({'Antithetic' if antithetic else 'Naive'})",
            "Price": f"{mc_price:.5f}",
            "Std Error": f"{mc_se:.5f}",
            "95% CI": f"[{mc_ci_lo:.4f}, {mc_ci_hi:.4f}]",
            "Time (s)": f"{mc_res['elapsed']:.3f}",
        })
        if bs_res:
            rows.append({
                "Method": "Black-Scholes (Analytical)",
                "Price": f"{bs_res['price']:.5f}",
                "Std Error": "—",
                "95% CI": "—",
                "Time (s)": f"{bs_res['elapsed']:.4f}",
            })
            rel_err = (mc_price - bs_res["price"]) / bs_res["price"] * 100
            rows.append({
                "Method": "Relative Error MC vs BS",
                "Price": f"{rel_err:+.3f}%",
                "Std Error": "—",
                "95% CI": "—",
                "Time (s)": "—",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Payoff diagram ──
        st.markdown(f"<h3>Payoff at Maturity</h3>", unsafe_allow_html=True)
        S_range = np.linspace(S0 * 0.5, S0 * 1.5, 300)
        payoff = np.array([option.pay_off(s) for s in S_range])
        intrinsic = payoff  # at maturity

        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.fill_between(S_range, payoff, alpha=0.15, color=RED)
        ax.plot(S_range, payoff, color=RED, linewidth=2, label="Payoff")
        ax.axvline(K, color=GOLD, linestyle="--", linewidth=1.2, label=f"Strike K={K:.0f}")
        ax.axvline(S0, color=GREEN, linestyle=":", linewidth=1.2, label=f"Spot S₀={S0:.0f}")
        ax.axhline(mc_price, color=WHITE, linestyle="--", linewidth=1,
                   alpha=0.5, label=f"MC Price={mc_price:.3f}")
        ax.set_xlabel("Spot at maturity")
        ax.set_ylabel("Payoff")
        ax.set_title(f"{cp} {ex} – K={K:.0f}, T={T_months}M")
        ax.legend()
        _apply_dark_style(fig)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("👆 Set parameters in the sidebar and click **Run Pricing**.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PATHS
# ══════════════════════════════════════════════════════════════════════════════

with tab_paths:
    st.markdown(f"<h2>Simulated GBM Paths</h2>", unsafe_allow_html=True)
    n_show = st.slider("Paths to display", 10, 200, 50, 10)

    with st.spinner("Simulating paths…"):
        t_arr, paths_arr = simulate_paths(S0, sigma, r, T_months, n_display=n_show, seed_val=seed_val or 42)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: paths
    ax = axes[0]
    cmap = plt.cm.get_cmap("RdYlBu", n_show)
    for i in range(min(n_show, paths_arr.shape[0])):
        ax.plot(t_arr, paths_arr[i], linewidth=0.6, alpha=0.55, color=cmap(i / n_show))
    ax.axhline(K, color=RED, linestyle="--", linewidth=1.5, label=f"Strike K={K:.0f}")
    ax.axhline(S0, color=GOLD, linestyle=":", linewidth=1.2, label=f"S₀={S0:.0f}")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Stock price")
    ax.set_title(f"{n_show} Monte Carlo paths  (σ={sigma:.0%}, r={r:.1%})")
    ax.legend()

    # Right: terminal distribution
    ax2 = axes[1]
    S_T = paths_arr[:, -1]
    ax2.hist(S_T, bins=40, color=RED, alpha=0.75, edgecolor=NAVY2, density=True)
    mu_gbm = S0 * np.exp(r * T_months / 12)
    ax2.axvline(mu_gbm, color=GOLD, linestyle="--", linewidth=1.5, label=f"E[S_T]={mu_gbm:.1f}")
    ax2.axvline(K, color=WHITE, linestyle=":", linewidth=1.2, label=f"Strike={K:.0f}")
    ax2.set_xlabel("S_T — terminal spot")
    ax2.set_ylabel("Density")
    ax2.set_title("Terminal spot distribution")
    ax2.legend()

    _apply_dark_style(fig)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Stats
    itm = np.mean(
        S_T > K if cp == "CALL" else S_T < K
    ) * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean S_T", f"{S_T.mean():.2f}")
    c2.metric("Std S_T", f"{S_T.std():.2f}")
    c3.metric("ITM paths", f"{itm:.1f}%")
    c4.metric("Min / Max", f"{S_T.min():.1f} / {S_T.max():.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GREEKS
# ══════════════════════════════════════════════════════════════════════════════

with tab_greeks:
    st.markdown(f"<h2>Monte Carlo Greeks (Finite Differences + CRN)</h2>", unsafe_allow_html=True)

    run_g = st.button("▶  Compute Greeks", use_container_width=False, key="btn_greeks")

    if run_g or "greeks_computed" in st.session_state:
        st.session_state["greeks_computed"] = True
        with st.spinner("Computing Greeks… (this may take ~10–20 s for N=50 k)"):
            t0 = time.perf_counter()
            ag: AllGreeks = compute_greeks(
                S0, K, sigma, r, div_a, T_months, cp, ex,
                num_paths, antithetic, seed_val,
            )
            elapsed_g = time.perf_counter() - t0

        st.success(f"Greeks computed in {elapsed_g:.1f}s")

        # ── Metric cards ──
        c1, c2, c3, c4, c5 = st.columns(5)
        cards = [
            (c1, "Delta  Δ", ag.delta),
            (c2, "Gamma  Γ", ag.gamma),
            (c3, "Vega  ν", ag.vega),
            (c4, "Theta  Θ", ag.theta),
            (c5, "Rho  ρ", ag.rho),
        ]
        for col, label, g in cards:
            lo, hi = g.ci95()
            col.metric(
                label,
                f"{g.value:.5f}",
                delta=f"±{1.96*g.se:.5f}",
            )

        # ── Table ──
        st.markdown(f"<h3>Full Greek Table</h3>", unsafe_allow_html=True)
        rows = []
        for label, g in [("Delta", ag.delta), ("Gamma", ag.gamma),
                          ("Vega", ag.vega), ("Theta", ag.theta), ("Rho", ag.rho)]:
            lo, hi = g.ci95()
            rows.append({
                "Greek": label,
                "Value": f"{g.value:.6f}",
                "Std Error": f"{g.se:.6f}",
                "95% CI Low": f"{lo:.6f}",
                "95% CI High": f"{hi:.6f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Bar chart ──
        st.markdown(f"<h3>Greeks (normalized display)</h3>", unsafe_allow_html=True)
        names = ["Delta", "Gamma", "Vega", "Theta", "Rho"]
        values = [ag.delta.value, ag.gamma.value, ag.vega.value, ag.theta.value, ag.rho.value]
        errors = [1.96 * g.se for g in [ag.delta, ag.gamma, ag.vega, ag.theta, ag.rho]]
        colors = [GREEN if v >= 0 else RED for v in values]

        fig, ax = plt.subplots(figsize=(8, 3.5))
        bars = ax.bar(names, values, color=colors, alpha=0.85, edgecolor=NAVY2, linewidth=0.5)
        ax.errorbar(names, values, yerr=errors, fmt="none", ecolor=WHITE,
                    elinewidth=1.5, capsize=5, capthick=1.5)
        ax.axhline(0, color=GREY, linewidth=0.8, linestyle="--")
        ax.set_ylabel("Value")
        ax.set_title(f"Greeks — {cp} {ex}  S₀={S0:.0f}  K={K:.0f}  σ={sigma:.0%}  T={T_months}M")
        _apply_dark_style(fig)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── BS comparison (European only) ──
        if ex == "EUROPEAN":
            st.markdown(f"<h3>MC Greeks vs Black-Scholes Analytics</h3>", unsafe_allow_html=True)
            from scipy.stats import norm as _norm
            import math

            T = T_months / 12
            S_eff = S0
            d1 = (math.log(S_eff / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            df = math.exp(-r * T)

            if cp == "CALL":
                bs_delta = _norm.cdf(d1)
                bs_gamma = _norm.pdf(d1) / (S0 * sigma * math.sqrt(T))
                bs_vega = S0 * _norm.pdf(d1) * math.sqrt(T)
                bs_theta = (-(S0 * _norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
                            - r * K * df * _norm.cdf(d2)) / 365
                bs_rho = K * T * df * _norm.cdf(d2)
            else:
                bs_delta = _norm.cdf(d1) - 1
                bs_gamma = _norm.pdf(d1) / (S0 * sigma * math.sqrt(T))
                bs_vega = S0 * _norm.pdf(d1) * math.sqrt(T)
                bs_theta = (-(S0 * _norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
                            + r * K * df * _norm.cdf(-d2)) / 365
                bs_rho = -K * T * df * _norm.cdf(-d2)

            cmp_rows = []
            for label, mc_val, bs_val in [
                ("Delta", ag.delta.value, bs_delta),
                ("Gamma", ag.gamma.value, bs_gamma),
                ("Vega",  ag.vega.value,  bs_vega),
                ("Theta", ag.theta.value, bs_theta),
                ("Rho",   ag.rho.value,   bs_rho),
            ]:
                diff = mc_val - bs_val
                rel = diff / abs(bs_val) * 100 if abs(bs_val) > 1e-10 else float("nan")
                cmp_rows.append({
                    "Greek": label,
                    "MC Value": f"{mc_val:.6f}",
                    "BS Value": f"{bs_val:.6f}",
                    "Diff": f"{diff:+.6f}",
                    "Rel Error": f"{rel:+.2f}%",
                })
            st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)
    else:
        st.info("👆 Click **Compute Greeks** to run finite-difference MC Greeks.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CONVERGENCE
# ══════════════════════════════════════════════════════════════════════════════

with tab_conv:
    st.markdown(f"<h2>MC Convergence Study</h2>", unsafe_allow_html=True)

    run_c = st.button("▶  Run Convergence", use_container_width=False, key="btn_conv")

    if run_c or "conv_computed" in st.session_state:
        st.session_state["conv_computed"] = True
        with st.spinner("Running convergence study (8 path counts)…"):
            df_conv, bs_ref = compute_convergence(
                S0, K, sigma, r, div_a, T_months, cp, ex, antithetic, seed_val, num_steps,
            )

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        # Left: price convergence
        ax = axes[0]
        ax.plot(df_conv["N"], df_conv["price"], color=RED, linewidth=2,
                marker="o", markersize=4, label="MC Price")
        ax.fill_between(
            df_conv["N"],
            df_conv["price"] - 1.96 * df_conv["se"],
            df_conv["price"] + 1.96 * df_conv["se"],
            alpha=0.2, color=RED, label="95% CI",
        )
        if bs_ref is not None:
            ax.axhline(bs_ref, color=GOLD, linestyle="--", linewidth=1.5,
                       label=f"BS reference = {bs_ref:.4f}")
        ax.set_xscale("log")
        ax.set_xlabel("Number of paths N  (log scale)")
        ax.set_ylabel("Estimated price")
        ax.set_title(f"Price convergence — {cp} {ex}")
        ax.legend()

        # Right: std error × sqrt(N) should be flat
        ax2 = axes[1]
        ax2.plot(df_conv["N"], df_conv["se"], color=GOLD, linewidth=2,
                 marker="s", markersize=4, label="Std Error")
        theory = df_conv["se"].iloc[-1] * np.sqrt(df_conv["N"].iloc[-1] / df_conv["N"])
        ax2.plot(df_conv["N"], theory, color=GREY, linestyle="--",
                 linewidth=1, label="O(1/√N) reference")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("N  (log scale)")
        ax2.set_ylabel("Std Error  (log scale)")
        ax2.set_title("Standard error vs N")
        ax2.legend()

        _apply_dark_style(fig)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Table
        df_display = df_conv.copy()
        df_display["95% CI"] = df_display.apply(
            lambda row: f"[{row['price']-1.96*row['se']:.4f}, {row['price']+1.96*row['se']:.4f}]",
            axis=1,
        )
        if bs_ref:
            df_display["Rel Error vs BS"] = df_display["price"].apply(
                lambda p: f"{(p - bs_ref)/bs_ref*100:+.3f}%"
            )
        df_display["price"] = df_display["price"].map("{:.5f}".format)
        df_display["se"] = df_display["se"].map("{:.5f}".format)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("👆 Click **Run Convergence** to compute prices for 8 path counts.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — STRIKE PROFILE
# ══════════════════════════════════════════════════════════════════════════════

with tab_smile:
    st.markdown(f"<h2>Price Profile vs Strike (σ=20%)</h2>", unsafe_allow_html=True)
    st.caption("Prices at fixed σ=20% across a range of strikes to visualize the moneyness profile.")

    run_s = st.button("▶  Run Strike Profile", use_container_width=False, key="btn_smile")

    if run_s or "smile_computed" in st.session_state:
        st.session_state["smile_computed"] = True
        with st.spinner("Computing across strikes…"):
            strikes, bs_prices, mc_prices = compute_smile(
                S0, T_months, r, div_a, ex, num_paths, antithetic, seed_val, num_steps, cp,
            )

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(strikes, mc_prices, color=RED, linewidth=2, marker="o",
                markersize=4, label="MC Price")
        if ex == "EUROPEAN" and any(b is not None for b in bs_prices):
            ax.plot(strikes, bs_prices, color=GOLD, linewidth=2,
                    linestyle="--", label="BS Price")
        ax.axvline(S0, color=GREEN, linestyle=":", linewidth=1.2, label=f"S₀={S0:.0f}")
        ax.axvline(K, color=WHITE, linestyle="--", linewidth=1.0, alpha=0.6,
                   label=f"Current K={K:.0f}")
        # Moneyness zones
        ax.axvspan(min(strikes), S0, alpha=0.05, color=RED,
                   label="OTM (Call) / ITM (Put)" if cp == "CALL" else "ITM (Put)")
        ax.axvspan(S0, max(strikes), alpha=0.05, color=GREEN)
        ax.set_xlabel("Strike  K")
        ax.set_ylabel("Option price")
        ax.set_title(f"{cp} {ex} price profile — S₀={S0:.0f}  T={T_months}M  σ=20%  r={r:.1%}")
        ax.legend()
        _apply_dark_style(fig)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("👆 Click **Run Strike Profile** to compute.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════

with tab_about:
    st.markdown(
        f"""
<h2>About this project</h2>

<div style="
    background:{NAVY2};
    border:1px solid #1E3A5F;
    border-left:4px solid {RED};
    border-radius:8px;
    padding:20px 24px;
    font-size:0.92rem;
    line-height:1.7;
">

<h3>Overview</h3>
<p>This dashboard demonstrates option pricing across three complementary approaches:</p>
<ul>
  <li><b style="color:{RED_LIGHT}">Black-Scholes</b> — closed-form analytical pricer (European only, with discrete dividend adjustment via the escrow approximation)</li>
  <li><b style="color:{RED_LIGHT}">Monte Carlo</b> — vectorized GBM simulation with optional antithetic variates for European options</li>
  <li><b style="color:{RED_LIGHT}">Longstaff-Schwartz</b> — least-squares Monte Carlo for American exercise, using polynomial regression on the continuation value</li>
</ul>

<h3>Greeks</h3>
<p>All Greeks are estimated via <b>central finite differences with Common Random Numbers (CRN)</b>: both the bumped and base pricings use the same random seed, 
dramatically reducing estimator variance.</p>

<h3>Package structure</h3>
<pre style="background:#0A1628;padding:10px;border-radius:6px;font-size:0.82rem">
src/
├── instruments/   market.py · option_trade.py
├── models/        brownian_motion.py
├── pricing/       black_scholes.py · monte_carlo_model.py · greeks.py · regression.py · pricing_result.py
├── studies/       convergence.py
└── benchmarks/    trinomial_tree/
</pre>

<h3>References</h3>
<ul>
  <li>Longstaff & Schwartz (2001) — <i>Valuing American Options by Simulation</i></li>
  <li>Glasserman (2004) — <i>Monte Carlo Methods in Financial Engineering</i></li>
  <li>Black & Scholes (1973)</li>
</ul>

</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Université", "Paris Dauphine PSL")
    col2.metric("Année", "2025 – 2026")
    col3.metric("Python", "3.11")
