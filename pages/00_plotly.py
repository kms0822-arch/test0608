import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global Market Cap TOP 10",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design Tokens ─────────────────────────────────────────────────────────────
DARK_BG      = "#0D1117"
PANEL_BG     = "#161B22"
BORDER       = "#21262D"
ACCENT       = "#58A6FF"      # electric blue — primary
GREEN        = "#3FB950"
RED          = "#F85149"
TEXT_PRIMARY = "#E6EDF3"
TEXT_MUTED   = "#8B949E"
GRID_COLOR   = "#21262D"

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
  }}

  /* hide Streamlit chrome */
  #MainMenu, footer, header {{ visibility: hidden; }}

  /* sidebar */
  [data-testid="stSidebar"] {{
    background: {PANEL_BG};
    border-right: 1px solid {BORDER};
  }}
  [data-testid="stSidebar"] * {{ color: {TEXT_PRIMARY} !important; }}

  /* metric cards */
  [data-testid="metric-container"] {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 16px 20px;
  }}
  [data-testid="stMetricLabel"]  {{ color: {TEXT_MUTED} !important; font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; }}
  [data-testid="stMetricValue"]  {{ color: {TEXT_PRIMARY} !important; font-weight: 700; font-size: 1.4rem; }}
  [data-testid="stMetricDelta"]  {{ font-size: 0.85rem !important; }}

  /* page title */
  .dash-title {{
    font-size: 1.7rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: {TEXT_PRIMARY};
    margin-bottom: 0;
  }}
  .dash-sub {{
    font-size: 0.85rem;
    color: {TEXT_MUTED};
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
  }}

  /* section labels */
  .section-label {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {ACCENT};
    margin-bottom: 12px;
  }}

  /* table */
  .styled-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }}
  .styled-table th {{
    background: {BORDER};
    color: {TEXT_MUTED};
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 10px 14px;
    text-align: left;
  }}
  .styled-table td {{
    padding: 10px 14px;
    border-bottom: 1px solid {BORDER};
    color: {TEXT_PRIMARY};
  }}
  .styled-table tr:last-child td {{ border-bottom: none; }}
  .styled-table tr:hover td {{ background: {BORDER}; }}
  .positive {{ color: {GREEN} !important; font-weight: 600; }}
  .negative {{ color: {RED}   !important; font-weight: 600; }}
  .mono {{ font-family: 'JetBrains Mono', monospace; }}

  /* stButton */
  div.stButton > button {{
    background: {ACCENT};
    color: #0D1117;
    border: none;
    font-weight: 600;
    border-radius: 6px;
  }}
  div.stButton > button:hover {{ opacity: 0.85; }}

  /* Plotly chart background to match theme */
  .js-plotly-plot .plotly {{ background: transparent !important; }}
</style>
""", unsafe_allow_html=True)

# ── Data ─────────────────────────────────────────────────────────────────────
TOP10 = {
    "AAPL":  {"name": "Apple",      "sector": "Technology"},
    "MSFT":  {"name": "Microsoft",  "sector": "Technology"},
    "NVDA":  {"name": "NVIDIA",     "sector": "Semiconductors"},
    "AMZN":  {"name": "Amazon",     "sector": "E-Commerce"},
    "GOOGL": {"name": "Alphabet",   "sector": "Technology"},
    "META":  {"name": "Meta",       "sector": "Social Media"},
    "TSLA":  {"name": "Tesla",      "sector": "Automotive"},
    "AVGO":  {"name": "Broadcom",   "sector": "Semiconductors"},
    "BRK-B": {"name": "Berkshire",  "sector": "Finance"},
    "JPM":   {"name": "JPMorgan",   "sector": "Finance"},
}

SECTOR_COLORS = {
    "Technology":    ACCENT,
    "Semiconductors":"#A371F7",
    "E-Commerce":    "#FFA657",
    "Social Media":  "#FF7B72",
    "Automotive":    "#3FB950",
    "Finance":       "#F0C27F",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def plotly_layout(title="", height=480):
    return dict(
        title=dict(text=title, font=dict(size=13, color=TEXT_MUTED), x=0, pad=dict(l=4)),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=TEXT_PRIMARY),
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=BORDER, showline=True, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=BORDER, showline=True, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=1,
                    font=dict(size=11)),
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
    )

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_prices(tickers: list[str], period="1y") -> pd.DataFrame:
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    close = raw["Close"] if "Close" in raw.columns else raw.xs("Close", axis=1, level=0)
    return close.dropna(how="all")

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_info(tickers: list[str]) -> dict:
    info = {}
    for t in tickers:
        try:
            d = yf.Ticker(t).fast_info
            info[t] = {
                "market_cap": getattr(d, "market_cap", None),
                "price":      getattr(d, "last_price", None),
            }
        except Exception:
            info[t] = {"market_cap": None, "price": None}
    return info

def fmt_mcap(v):
    if v is None: return "—"
    if v >= 1e12: return f"${v/1e12:.2f}T"
    if v >= 1e9:  return f"${v/1e9:.1f}B"
    return f"${v:,.0f}"

def pct_change_series(s: pd.Series) -> pd.Series:
    return (s / s.iloc[0] - 1) * 100

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    st.markdown("---")

    period_map = {"1개월": "1mo", "3개월": "3mo", "6개월": "6mo", "1년": "1y"}
    period_label = st.selectbox("📅 기간", list(period_map.keys()), index=3)
    period = period_map[period_label]

    chart_type = st.selectbox("📊 차트 유형", ["정규화 수익률", "캔들스틱", "히트맵"])

    selected = st.multiselect(
        "🔍 종목 선택",
        list(TOP10.keys()),
        default=list(TOP10.keys()),
        format_func=lambda t: f"{TOP10[t]['name']} ({t})",
    )
    if not selected:
        selected = list(TOP10.keys())

    st.markdown("---")
    if st.button("🔄 데이터 새로고침"):
        st.cache_data.clear()
        st.rerun()

    st.markdown(f"""
    <div style='color:{TEXT_MUTED}; font-size:0.75rem; margin-top:16px;'>
      데이터: Yahoo Finance<br>
      갱신: 30분 캐시<br>
      기준: 글로벌 시가총액 TOP 10
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='margin-bottom:24px;'>
  <div class='dash-title'>🌐 글로벌 시가총액 TOP 10 대시보드</div>
  <div class='dash-sub'>as of {datetime.now().strftime('%Y-%m-%d %H:%M')} KST &nbsp;·&nbsp; period: {period_label}</div>
</div>
""", unsafe_allow_html=True)

# ── Fetch ─────────────────────────────────────────────────────────────────────
with st.spinner("📡 Yahoo Finance에서 데이터를 불러오는 중..."):
    prices_df = fetch_prices(selected, period=period)
    info_dict = fetch_info(selected)

# filter to only tickers that actually loaded
valid = [t for t in selected if t in prices_df.columns and not prices_df[t].dropna().empty]
prices_df = prices_df[valid]

# ── KPI Row ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">실시간 스냅샷</div>', unsafe_allow_html=True)
cols = st.columns(min(len(valid), 5))
for i, t in enumerate(valid[:5]):
    s = prices_df[t].dropna()
    ret = (s.iloc[-1] / s.iloc[0] - 1) * 100
    info = info_dict.get(t, {})
    with cols[i]:
        st.metric(
            label=f"{TOP10[t]['name']} ({t})",
            value=f"${s.iloc[-1]:.2f}",
            delta=f"{ret:+.2f}% ({period_label})",
        )

cols2 = st.columns(min(len(valid[5:]), 5))
for i, t in enumerate(valid[5:]):
    s = prices_df[t].dropna()
    ret = (s.iloc[-1] / s.iloc[0] - 1) * 100
    with cols2[i]:
        st.metric(
            label=f"{TOP10[t]['name']} ({t})",
            value=f"${s.iloc[-1]:.2f}",
            delta=f"{ret:+.2f}% ({period_label})",
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Main Chart ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">주가 성과</div>', unsafe_allow_html=True)

if chart_type == "정규화 수익률":
    fig = go.Figure()
    for t in valid:
        s = prices_df[t].dropna()
        norm = pct_change_series(s)
        color = SECTOR_COLORS.get(TOP10[t]["sector"], ACCENT)
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm.values,
            name=f"{TOP10[t]['name']}",
            mode="lines",
            line=dict(width=2, color=color),
            hovertemplate=f"<b>{TOP10[t]['name']}</b><br>%{{x|%Y-%m-%d}}<br>수익률: %{{y:+.2f}}%<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dot", line_color=TEXT_MUTED, line_width=1)
    fig.update_layout(**plotly_layout("시작점 기준 정규화 수익률 (%)", height=500))
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "캔들스틱":
    tick = st.selectbox("종목 선택", valid, format_func=lambda t: f"{TOP10[t]['name']} ({t})")
    raw = yf.download(tick, period=period, auto_adjust=True, progress=False)
    if not raw.empty:
        fig = go.Figure(go.Candlestick(
            x=raw.index,
            open=raw["Open"].squeeze(),
            high=raw["High"].squeeze(),
            close=raw["Close"].squeeze(),
            low=raw["Low"].squeeze(),
            increasing_line_color=GREEN, decreasing_line_color=RED,
            name=tick,
        ))
        fig.update_layout(**plotly_layout(f"{TOP10[tick]['name']} 캔들스틱", height=500))
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

else:  # 히트맵
    # weekly returns heatmap
    weekly = prices_df[valid].resample("W").last().pct_change().dropna() * 100
    weekly.index = weekly.index.strftime("%Y-%m-%d")
    labels = [TOP10[t]["name"] for t in valid]
    fig = go.Figure(go.Heatmap(
        z=weekly[valid].T.values,
        x=weekly.index.tolist(),
        y=labels,
        colorscale=[[0, RED], [0.5, "#1C2128"], [1, GREEN]],
        zmid=0,
        text=weekly[valid].T.round(1).astype(str).values,
        texttemplate="%{text}%",
        hovertemplate="<b>%{y}</b><br>주간: %{x}<br>수익률: %{z:.2f}%<extra></extra>",
        colorbar=dict(title="주간 수익률 (%)", ticksuffix="%"),
    ))
    fig.update_layout(**plotly_layout("주간 수익률 히트맵", height=420))
    st.plotly_chart(fig, use_container_width=True)

# ── Bar Chart: Period Return ──────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
c1, c2 = st.columns([3, 2])

with c1:
    st.markdown('<div class="section-label">기간 수익률 비교</div>', unsafe_allow_html=True)
    rets = {}
    for t in valid:
        s = prices_df[t].dropna()
        rets[t] = (s.iloc[-1] / s.iloc[0] - 1) * 100
    rets_s = pd.Series(rets).sort_values(ascending=True)
    colors = [GREEN if v >= 0 else RED for v in rets_s.values]

    fig2 = go.Figure(go.Bar(
        x=rets_s.values,
        y=[TOP10[t]["name"] for t in rets_s.index],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in rets_s.values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>수익률: %{x:+.2f}%<extra></extra>",
    ))
    fig2.add_vline(x=0, line_color=TEXT_MUTED, line_width=1)
    fig2.update_layout(**plotly_layout("", height=360))
    fig2.update_xaxes(ticksuffix="%")
    st.plotly_chart(fig2, use_container_width=True)

with c2:
    st.markdown('<div class="section-label">시가총액 현황</div>', unsafe_allow_html=True)
    mcap_data = {
        t: info_dict[t]["market_cap"]
        for t in valid if info_dict[t]["market_cap"]
    }
    if mcap_data:
        mc_s = pd.Series(mcap_data).sort_values(ascending=False)
        sector_colors_list = [SECTOR_COLORS.get(TOP10[t]["sector"], ACCENT) for t in mc_s.index]
        fig3 = go.Figure(go.Bar(
            x=[TOP10[t]["name"] for t in mc_s.index],
            y=mc_s.values / 1e12,
            marker_color=sector_colors_list,
            text=[fmt_mcap(v) for v in mc_s.values],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>시가총액: %{text}<extra></extra>",
        ))
        fig3.update_layout(**plotly_layout("단위: 조 달러 (Trillion $)", height=360))
        fig3.update_yaxes(ticksuffix="T")
        st.plotly_chart(fig3, use_container_width=True)

# ── Detail Table ──────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">종목 상세 현황</div>', unsafe_allow_html=True)

rows = []
for t in valid:
    s = prices_df[t].dropna()
    ret_period  = (s.iloc[-1] / s.iloc[0] - 1) * 100
    high_52 = s.max()
    low_52  = s.min()
    from_high = (s.iloc[-1] / high_52 - 1) * 100
    info = info_dict.get(t, {})
    rows.append({
        "종목": f"{TOP10[t]['name']} <span class='mono' style='color:{TEXT_MUTED};font-size:0.8em;'>({t})</span>",
        "섹터": TOP10[t]["sector"],
        "현재가": f"<span class='mono'>${s.iloc[-1]:.2f}</span>",
        f"{period_label} 수익률": f"<span class='{'positive' if ret_period>=0 else 'negative'}'>{ret_period:+.2f}%</span>",
        "52주 고가": f"<span class='mono'>${high_52:.2f}</span>",
        "52주 저가": f"<span class='mono'>${low_52:.2f}</span>",
        "고점 대비": f"<span class='{'positive' if from_high>=0 else 'negative'}'>{from_high:.1f}%</span>",
        "시가총액": fmt_mcap(info.get("market_cap")),
    })

df_table = pd.DataFrame(rows)
header = "".join(f"<th>{c}</th>" for c in df_table.columns)
body = ""
for _, row in df_table.iterrows():
    cells = "".join(f"<td>{v}</td>" for v in row.values)
    body += f"<tr>{cells}</tr>"

st.markdown(f"""
<div style='background:{PANEL_BG}; border:1px solid {BORDER}; border-radius:10px; overflow:auto; padding:4px;'>
<table class='styled-table'>
  <thead><tr>{header}</tr></thead>
  <tbody>{body}</tbody>
</table>
</div>
""", unsafe_allow_html=True)

# ── Correlation Heatmap ───────────────────────────────────────────────────────
if len(valid) > 2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">수익률 상관관계</div>', unsafe_allow_html=True)

    corr = prices_df[valid].pct_change().dropna().corr()
    labels = [TOP10[t]["name"] for t in valid]
    fig4 = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels, y=labels,
        colorscale=[[0, "#F85149"], [0.5, PANEL_BG], [1, "#3FB950"]],
        zmin=-1, zmax=1, zmid=0,
        text=corr.round(2).values,
        texttemplate="%{text}",
        hovertemplate="<b>%{y} × %{x}</b><br>상관계수: %{z:.3f}<extra></extra>",
        colorbar=dict(title="상관계수"),
    ))
    fig4.update_layout(**plotly_layout("일간 수익률 기반 상관관계 매트릭스", height=440))
    st.plotly_chart(fig4, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='margin-top:48px; padding-top:20px; border-top:1px solid {BORDER};
            color:{TEXT_MUTED}; font-size:0.75rem; text-align:center;'>
  데이터 출처: Yahoo Finance via yfinance &nbsp;|&nbsp;
  본 대시보드는 정보 제공 목적이며 투자 조언이 아닙니다. &nbsp;|&nbsp;
  Built with Streamlit + Plotly
</div>
""", unsafe_allow_html=True)
