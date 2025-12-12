# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import base64
import mimetypes
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# 기본 설정
# ============================================================
st.set_page_config(
    page_title="브리즈번 수질 알리미",
    page_icon=":droplet:",
    layout="wide",
)

# ============================================================
# 데이터 로드
# ============================================================
@st.cache_data
def get_water_data():
    DATA_FILENAME = Path(__file__).parent / "data" / "df_final.csv"
    if not DATA_FILENAME.exists():
        st.error(f"데이터 파일을 찾을 수 없습니다: {DATA_FILENAME}")
        return pd.DataFrame()
    df = pd.read_csv(DATA_FILENAME)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["date"] = df["Timestamp"].dt.date
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


@st.cache_data
def load_future_forecast():
    path = Path(__file__).parent / "data" / "future_week_forecast.csv"
    if not path.exists():
        return None
    df_fore = pd.read_csv(path, parse_dates=["Timestamp"])
    if "Forecast_Chlorophyll_Kalman" not in df_fore.columns:
        return None
    df_fore = df_fore.sort_values("Timestamp").reset_index(drop=True)
    return df_fore


df = get_water_data()
forecast_df = load_future_forecast()

# ============================================================
# 도메인 헬퍼
# ============================================================
def classify_chl(value: float):
    if pd.isna(value):
        return "정보 부족", "⚪", "#9ca3af", "데이터가 부족해 정확한 상태 진단이 어렵습니다."
    if value < 4:
        return "좋음", "🟢", "#22c55e", "평상 수준으로, 산책·레저 활동에 비교적 안전한 상태입니다."
    if value < 8:
        return "주의", "🟡", "#eab308", "조류(녹조) 농도가 다소 높아진 상태입니다. 기상·강우에 따라 변동이 클 수 있습니다."
    return "위험", "🔴", "#ef4444", "조류(녹조) 농도가 높은 편입니다. 레저 활동 전 공식 안내를 꼭 확인해 주세요."


def get_last_valid(df_local: pd.DataFrame, col: str):
    if df_local is None or df_local.empty:
        return np.nan
    if col not in df_local.columns:
        return np.nan
    return df_local[col].dropna().iloc[-1] if df_local[col].notna().any() else np.nan


def add_risk_bands_plotly(fig, y_max: float):
    """Plotly 그래프에 위험 구간 밴드(0–4, 4–8, 8+) 추가."""
    fig.add_hrect(y0=0, y1=4, line_width=0, fillcolor="#22c55e", opacity=0.12)
    fig.add_hrect(y0=4, y1=8, line_width=0, fillcolor="#eab308", opacity=0.18)
    fig.add_hrect(y0=8, y1=y_max, line_width=0, fillcolor="#ef4444", opacity=0.12)
    fig.add_hline(y=4, line_dash="dot", line_color="#eab308", line_width=1)
    fig.add_hline(y=8, line_dash="dot", line_color="#ef4444", line_width=1)


def build_activity_recommendation(chl, temp, turb, label):
    """조류/수온/탁도 + 등급으로 오늘의 활동 추천 멘트 생성."""
    if any(pd.isna(x) for x in [chl, temp, turb]):
        return (
            "데이터 부족",
            "#9ca3af",
            "센서 데이터가 충분하지 않아 오늘의 활동을 정확히 추천하기 어렵습니다. "
            "현장 안내판·공식 공지를 함께 확인해 주세요.",
        )

    if label == "좋음" and 18 <= temp <= 26 and turb < 50:
        color = "#22c55e"
        title = "레저 활동하기 좋은 날"
        msg = (
            f"조류 농도 {chl:.1f} µg/L, 수온 {temp:.1f} °C, 탁도 {turb:.1f} NTU 수준으로 "
            "카약·패들보드 등 가벼운 수상 레저와 물가 산책을 즐기기 좋습니다. "
            "어린이 물놀이는 항상 보호자와 함께해 주세요."
        )
    elif label == "위험" or turb >= 80:
        color = "#ef4444"
        title = "물놀이 자제 권고"
        msg = (
            f"조류 농도 {chl:.1f} µg/L로 높은 편이며, 탁도 {turb:.1f} NTU 수준입니다. "
            "수영·튜브 등 직접 물에 들어가는 활동은 가급적 피하는 것이 좋습니다. "
            "강 주변 산책이나 조망 위주의 활동을 추천드립니다."
        )
    else:
        color = "#eab308"
        title = "가벼운 활동 권장 (주의)"
        msg = (
            f"조류 농도 {chl:.1f} µg/L, 수온 {temp:.1f} °C 수준으로 일부 시간대에 조류가 다소 높을 수 있습니다. "
            "카약·보트 등은 가능하지만, 물과의 직접 접촉은 줄이고 샤워 등 위생 관리를 신경 써 주세요."
        )

    return title, color, msg


# ============================================================
# 배경 이미지 + 상태 아이콘
# ============================================================
STATIC_DIR = Path(__file__).parent / "static"
img_good = STATIC_DIR / "bg_good.jpg"
img_warning = STATIC_DIR / "bg_warning.jpg"
img_danger = STATIC_DIR / "bg_danger.jpg"
img_unknown = STATIC_DIR / "bg_unknown.jpg"

icon_good = STATIC_DIR / "icon_good.png"
icon_warning = STATIC_DIR / "icon_warning.png"
icon_danger = STATIC_DIR / "icon_danger.png"
icon_unknown = STATIC_DIR / "icon_unknown.png"


def get_base64_image(path: Path):
    if not path.exists():
        return None
    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


# ============================================================
# 기본 정보 계산 + 지표 조회 날짜 결정
# ============================================================
if "Timestamp" in df.columns and not df.empty:
    df = df.sort_values("Timestamp")
    latest_row = df.iloc[-1]
    latest_time = latest_row["Timestamp"]
    today_date = latest_time.date()
else:
    latest_row = df.iloc[-1] if not df.empty else None
    latest_time = (
        latest_row["Timestamp"]
        if latest_row is not None and "Timestamp" in latest_row.index
        else None
    )
    today_date = df["date"].iloc[-1] if "date" in df.columns and not df.empty else None

# 지표 조회 날짜 기본값/선택값
if not df.empty and "date" in df.columns:
    available_dates = sorted(df["date"].unique())
    default_date = today_date or available_dates[-1]

    if "metric_date" in st.session_state:
        sd = st.session_state["metric_date"]
        if isinstance(sd, pd.Timestamp):
            sd = sd.date()
        elif isinstance(sd, datetime.datetime):
            sd = sd.date()
        if sd < available_dates[0] or sd > available_dates[-1]:
            sd = default_date
        selected_date = sd
    else:
        selected_date = default_date
else:
    available_dates = None
    selected_date = today_date

# 선택 날짜 기준 데이터프레임
if not df.empty and "date" in df.columns and selected_date is not None:
    sel_df = df[df["date"] == selected_date]
else:
    sel_df = df.copy()

# 선택 날짜 기준 현재값
sel_chl = get_last_valid(sel_df, "Chlorophyll_Kalman")
sel_temp = get_last_valid(sel_df, "Temperature_Kalman")
sel_turb = get_last_valid(sel_df, "Turbidity_Kalman")
sel_do = get_last_valid(sel_df, "Dissolved Oxygen_Kalman")

# 선택 날짜 기준 마지막 시각
if not sel_df.empty and "Timestamp" in sel_df.columns:
    sel_time = sel_df["Timestamp"].iloc[-1]
else:
    sel_time = latest_time

# 선택 날짜 기준 범위 텍스트
if (
    "Chlorophyll_Kalman" in sel_df.columns
    and not sel_df["Chlorophyll_Kalman"].dropna().empty
):
    sel_min = sel_df["Chlorophyll_Kalman"].min()
    sel_max = sel_df["Chlorophyll_Kalman"].max()
    if today_date is not None and selected_date == today_date:
        hero_range_text = f"오늘 범위: {sel_min:.1f} ~ {sel_max:.1f} µg/L"
    else:
        hero_range_text = (
            f"{selected_date.strftime('%m/%d')} 범위: {sel_min:.1f} ~ {sel_max:.1f} µg/L"
        )
else:
    hero_range_text = "범위: 데이터 없음"

# 선택 날짜 기준 등급 → 배경/아이콘에 사용
hero_label, hero_emoji, hero_color, _ = classify_chl(sel_chl)

# 배경 이미지
if hero_label == "좋음":
    chosen_img = img_good
elif hero_label == "주의":
    chosen_img = img_warning
elif hero_label == "위험":
    chosen_img = img_danger
else:
    chosen_img = img_unknown

bg_data_uri = get_base64_image(chosen_img)
bg_css_url = bg_data_uri if bg_data_uri else None

# TODAY 카드용 상태 아이콘
if hero_label == "좋음":
    hero_icon_path = icon_good
elif hero_label == "주의":
    hero_icon_path = icon_warning
elif hero_label == "위험":
    hero_icon_path = icon_danger
else:
    hero_icon_path = icon_unknown

hero_icon_uri = get_base64_image(hero_icon_path) if hero_icon_path is not None else None

# ============================================================
# CSS 스타일
# ============================================================
css_block = "<style>"

if bg_css_url:
    css_block += f"""
.stApp {{
    background-image: url("{bg_css_url}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    color: #e5e7eb;
}}
"""
else:
    css_block += """
.stApp {
    background-color: #020617;
    color: #e5e7eb;
}
"""

css_block += """
.block-container {
    padding-top: 2.8rem;
    padding-bottom: 2rem;
    padding-left: 1.4rem;
    padding-right: 1.4rem;
}
@media (min-width: 1200px) {
  .block-container {
      padding-left: 5rem;
      padding-right: 5rem;
  }
}

/* 공통 카드 */
.card {
    background-color: rgba(15, 23, 42, 0.75);
    border-radius: 1.4rem;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 18px 40px rgba(0,0,0,0.45);
    backdrop-filter: blur(18px);
}

/* Plotly 차트 카드 스타일 */
div[data-testid="stPlotlyChart"] {
    background-color: rgba(15, 23, 42, 0.75);
    border-radius: 1.4rem;
    padding: 0.8rem 1.0rem 1.0rem 1.0rem;
    box-shadow: 0 18px 40px rgba(0,0,0,0.45);
    backdrop-filter: blur(18px);
}

/* 메인 타이틀 */
.main-title {
    font-size: clamp(24px, 2.6vw, 32px);
    font-weight: 800;
    color: #f9fafb;
    margin-bottom: 0.15rem;
}
.sub-title {
    font-size: 13px;
    opacity: 0.85;
    margin-bottom: 0.7rem;
}
.tag-pill {
    display: inline-block;
    padding: 0.12rem 0.55rem;
    border-radius: 999px;
    font-size: 0.7rem;
    margin-right: 0.25rem;
    background-color: rgba(15, 23, 42, 0.75);
    color: #e5e7eb;
    border: 1px solid rgba(148, 163, 184, 0.4);
}

/* TODAY 카드 */
.hero-card {
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 1.8rem 1.4rem 1.6rem 1.4rem;
}
.hero-title {
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: 0.8;
}
.hero-location {
    font-size: 1.1rem;
    margin-top: 0.25rem;
    font-weight: 600;
}
.hero-main-row {
    display: flex;
    align-items: flex-end;
    gap: 0.2rem;
    margin-top: 0.55rem;
}
.hero-main-value {
    font-size: clamp(3.0rem, 7vw, 3.8rem);
    font-weight: 800;
    line-height: 1.05;
}
.hero-main-unit {
    font-size: 1.1rem;
    opacity: 0.85;
    margin-bottom: 0.35rem;
}
.hero-condition {
    font-size: 1.05rem;
    margin-top: 0.25rem;
}
.hero-range {
    font-size: 0.78rem;
    margin-top: 0.4rem;
    opacity: 0.9;
}
.hero-grade-guide {
    font-size: 0.75rem;
    opacity: 0.85;
    margin-top: 0.45rem;
}

/* 상태 아이콘 */
.hero-icon {
    width: 72px;
    margin-top: 0.6rem;
    margin-bottom: 0.25rem;
}

/* 현재 주요 지표 카드 */
.chip-box {
    border-radius: 1.0rem;
    padding: 0.7rem 0.9rem;
    background-color: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.4);
    font-size: 0.78rem;
    margin-bottom: 0.45rem;
    text-align: center;
}
.chip-label {
    opacity: 0.7;
    font-size: 0.76rem;
}
.chip-value {
    font-size: 1.02rem;
    font-weight: 600;
    margin-top: 0.2rem;
}
.small-title {
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 0.2rem;
    margin-top: 0.4rem;
}

/* 오늘의 추천 활동 카드 */
.recommend-card {
    margin-top: 0.45rem;
    border-radius: 1.0rem;
    padding: 0.85rem 0.95rem;
    background-color: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.5);
    font-size: 0.8rem;
}
.recommend-title {
    font-size: 0.86rem;
    font-weight: 600;
    margin-bottom: 0.25rem;
}
.recommend-body {
    font-size: 0.78rem;
    line-height: 1.5;
}

/* 섹션 타이틀 */
.section-title {
    font-size: 1.3rem;
    font-weight: 600;
    margin-top: 1.4rem;
    margin-bottom: 0.35rem;
}
.info-text {
    font-size: 0.8rem;
    opacity: 0.82;
}

/* 주간 예보 카드 */
.week-card-header {
    display: flex;
    align-items: flex-start;        /* ✅ 제목/기간 텍스트 상단 정렬 */
    justify-content: space-between;
    margin-bottom: 0.45rem;
    font-size: 0.86rem;
}
.week-card-title {
    font-size: 1.05rem;            /* ✅ 1) 제목 크기 조절 */
    font-weight: 700;
    transform: translateY(-2px);   /* ✅ 1) 제목 위치 미세조정(위로) */
}
.week-subtitle {
    font-size: 0.76rem;
    opacity: 0.85;
}

.week-rows {
    margin-top: 0.25rem;
}

/* ✅ 2) 평균 열 추가로 grid 컬럼 6개로 변경 */
.week-header-row {
    display: grid;
    grid-template-columns: 1.5fr 1.6fr 0.9fr 0.9fr 4.0fr 0.9fr;
    column-gap: 0.45rem;
    font-size: 0.76rem;
    opacity: 0.9;
    padding-bottom: 0.15rem;
    border-bottom: 1px solid rgba(148,163,184,0.35);
    margin-bottom: 0.15rem;
    text-align: center;
}

.week-row {
    display: grid;
    grid-template-columns: 1.5fr 1.6fr 0.9fr 0.9fr 4.0fr 0.9fr;
    align-items: center;
    column-gap: 0.45rem;
    padding: 0.25rem 0;
    font-size: 0.82rem;
    text-align: center;
}

.week-day {
    font-weight: 500;
}
.week-status {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.25rem;
}
.week-emoji {
    font-size: 1.0rem;
}
.week-status-text {
    font-size: 0.78rem;
    opacity: 0.9;
}
.week-mean,
.week-min,
.week-max {
    font-variant-numeric: tabular-nums;
    opacity: 0.9;
}

.week-range-track {
    position: relative;
    height: 0.42rem;
    border-radius: 999px;
    background-color: rgba(148, 163, 184, 0.3);
    overflow: hidden;
}
.week-range-bar {
    position: absolute;
    top: 0;
    bottom: 0;
    border-radius: 999px;
}

/* 평균값 빨간 굵은 바 */
.week-mean-marker {
    position: absolute;
    top: -0.14rem;
    width: 4px;
    height: 0.70rem;
    background-color: #ef4444;
    border-radius: 999px;
}

/* 지표 조회 날짜 위젯 스타일 */
div[data-testid="stDateInput"] label {
    color: #f9fafb !important;
    font-size: 0.78rem;
    margin-bottom: 0.1rem;
}
div[data-testid="stDateInput"] > div {
    background-color: rgba(15, 23, 42, 0.75) !important;
    border-radius: 0.75rem;
    border: 1px solid rgba(148, 163, 184, 0.5);
}
div[data-testid="stDateInput"] input {
    color: #000000 !important;
    background-color: #f9fafb !important;
    caret-color: #000000 !important;
}
div[data-testid="stDateInput"] svg {
    color: #f9fafb !important;
}

/* 데이터 테이블 섹션 */
.expander-text {
    font-size: 0.8rem;
    opacity: 0.85;
}

/* expander 제목/다운로드 버튼 글자색 검정 */
div[data-testid="stExpander"] summary {
    color: #000000 !important;
}
div[data-testid="stDownloadButton"] button {
    color: #000000 !important;
}

/* ✅ expander 위쪽 간격 추가 */
div[data-testid="stExpander"] {
    margin-top: 1.2rem;
}

/* ✅ expander 내용 전체를 카드처럼 */
div[data-testid="stExpanderDetails"] {
    background-color: rgba(15, 23, 42, 0.75);
    border-radius: 1.4rem;
    padding: 1.0rem 1.0rem 1.1rem 1.0rem;
    box-shadow: 0 18px 40px rgba(0,0,0,0.45);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(148, 163, 184, 0.4);
}

/* ✅ expander 안 Plotly는 '카드 중복' 제거 */
div[data-testid="stExpanderDetails"] div[data-testid="stPlotlyChart"] {
    background-color: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* (선택) expander 안 DataFrame 정리 */
div[data-testid="stExpanderDetails"] div[data-testid="stDataFrame"] {
    border-radius: 1.0rem;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.35);
}

/* ✅ 슬라이더/셀렉트 라벨 흰색 */
div[data-testid="stSlider"] label,
div[data-testid="stSelectbox"] label {
    color: #f9fafb !important;
}

</style>
"""

st.markdown(css_block, unsafe_allow_html=True)

# ============================================================
# 헤더
# ============================================================
st.markdown('<div class="main-title">브리즈번 수질 알리미</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">브리즈번 강(Colmslie Buoy) 수질을 날씨앱처럼 한눈에 확인하세요.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
<span class="tag-pill">실시간 센서</span>
<span class="tag-pill">조류(클로로필) 모니터링</span>
<span class="tag-pill">7일 예보</span>
<span class="tag-pill">시민용 안내</span>
""",
    unsafe_allow_html=True,
)
st.write("")

# ============================================================
# 1. 오늘의 브리즈번 강 상태
# ============================================================
col_hero_main, col_hero_side = st.columns([2, 1.4])

with col_hero_side:
    st.markdown('<div class="small-title">현재 주요 지표</div>', unsafe_allow_html=True)

    if not df.empty and "date" in df.columns and available_dates is not None:
        st.date_input(
            "지표 조회 날짜",
            value=selected_date,
            min_value=available_dates[0],
            max_value=available_dates[-1],
            key="metric_date",
        )
    else:
        st.write("데이터가 부족하여 날짜 선택이 어렵습니다.")

    c1, c2 = st.columns(2)
    with c1:
        temp_text = "–" if pd.isna(sel_temp) else f"{sel_temp:.1f} °C"
        st.markdown(
            f"""
<div class="chip-box">
  <div class="chip-label">수온</div>
  <div class="chip-value">{temp_text}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with c2:
        turb_text = "–" if pd.isna(sel_turb) else f"{sel_turb:.1f} NTU"
        st.markdown(
            f"""
<div class="chip-box">
  <div class="chip-label">탁도</div>
  <div class="chip-value">{turb_text}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    c3, c4 = st.columns(2)
    with c3:
        do_text = "–" if pd.isna(sel_do) else f"{sel_do:.1f} mg/L"
        st.markdown(
            f"""
<div class="chip-box">
  <div class="chip-label">용존산소</div>
  <div class="chip-value">{do_text}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with c4:
        time_txt = sel_time.strftime("%Y-%m-%d %H:%M") if sel_time is not None else "정보 없음"
        st.markdown(
            f"""
<div class="chip-box">
  <div class="chip-label">마지막 업데이트</div>
  <div class="chip-value">{time_txt}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    chl_label_for_rec, _, _, _ = classify_chl(sel_chl)
    rec_title, rec_color, rec_msg = build_activity_recommendation(sel_chl, sel_temp, sel_turb, chl_label_for_rec)

    st.markdown(
        f"""
<div class="recommend-card">
  <div class="recommend-title">
    <span style="color:{rec_color}; font-size:0.9rem;">●</span>
    오늘의 추천 활동
  </div>
  <div class="recommend-body">
    <b>{rec_title}</b><br/>
    {rec_msg}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

with col_hero_main:
    chl_text = "–" if pd.isna(sel_chl) else f"{sel_chl:.1f}"
    icon_html = f'<img class="hero-icon" src="{hero_icon_uri}" />' if hero_icon_uri is not None else ""

    hero_html = f"""
<div class="card hero-card">
  <div class="hero-title">TODAY • BRISBANE RIVER • COLMSLIE</div>
  <div class="hero-location">브리즈번 강 조류 농도</div>

  {icon_html}

  <div class="hero-main-row">
    <span class="hero-main-value">{chl_text}</span>
    <span class="hero-main-unit">µg/L</span>
  </div>

  <div class="hero-condition" style="color:{hero_color};">
    {hero_emoji} {hero_label}
  </div>

  <div class="hero-range">{hero_range_text}</div>

  <div class="hero-grade-guide">
    🟢 0–4 : 양호&nbsp;&nbsp;&nbsp; 🟡 4–8 : 주의&nbsp;&nbsp;&nbsp; 🔴 8 이상 : 위험
  </div>
</div>
"""
    st.markdown(hero_html, unsafe_allow_html=True)

# ============================================================
# 2. 이번주 조류량 예측 + 위치 지도
# ============================================================
st.markdown('<div class="section-title">이번주 조류량 예측</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="info-text">예측 모델을 이용해 앞으로 7일 동안의 일별 조류 농도 범위(최저·최고)와 전체 추세를 함께 보여줍니다.</div>',
    unsafe_allow_html=True,
)

if forecast_df is None or forecast_df.empty:
    st.info("예측 파일(future_week_forecast.csv)을 찾을 수 없어, 주간 예보를 표시할 수 없습니다.")
else:
    df_fore = forecast_df.copy()
    df_fore["date"] = df_fore["Timestamp"].dt.date

    daily = (
        df_fore.groupby("date")["Forecast_Chlorophyll_Kalman"]
        .agg(["min", "max", "mean"])
        .reset_index()
    )
    daily = daily.sort_values("date").head(7)

    if daily.empty:
        st.warning("주간 예보 데이터가 없습니다.")
    else:
        global_min = daily["min"].min()
        global_max = daily["max"].max()
        denom = (
            global_max - global_min
            if pd.notna(global_min) and pd.notna(global_max) and global_max > global_min
            else None
        )

        weekdays_kr = ["월", "화", "수", "목", "금", "토", "일"]

        period_start = daily["date"].min()
        period_end = daily["date"].max()
        period_text = f"{period_start.strftime('%m월 %d일')} ~ {period_end.strftime('%m월 %d일')}"

        # 최대 예보 문구(날짜/수치 강조)
        max_info_text = None
        if forecast_df is not None and not forecast_df.empty:
            idxmax = forecast_df["Forecast_Chlorophyll_Kalman"].idxmax()
            max_future_value = forecast_df.loc[idxmax, "Forecast_Chlorophyll_Kalman"]
            max_future_time = forecast_df.loc[idxmax, "Timestamp"]

            if pd.notna(max_future_value) and pd.notna(max_future_time):
                lab, emo, _, _ = classify_chl(max_future_value)
                t_txt = max_future_time.strftime("%Y-%m-%d %H:%M")

                date_color = "#60a5fa"
                value_color = "#f97316"

                max_info_text = (
                    "가장 조류 농도가 높게 예보된 시점은 "
                    f"<span style='color:{date_color}; font-weight:700;'>{t_txt}</span>"
                    "이며, 예측값은 약 "
                    f"<span style='color:{value_color}; font-weight:800;'>{max_future_value:.1f} µg/L</span>"
                    f" ({emo} {lab}) 입니다."
                )

        st.markdown(
            '<div class="info-text" style="margin-top:0.4rem; margin-bottom:0.15rem;">라인 그래프 조회 일자</div>',
            unsafe_allow_html=True,
        )
        line_date_options = [None] + list(daily["date"])

        selected_line_date = st.selectbox(
            "",
            options=line_date_options,
            index=0,
            format_func=lambda d: "전체 기간" if d is None else d.strftime("%m/%d"),
            label_visibility="collapsed",
        )

        if selected_line_date is None:
            mask = (df_fore["date"] >= period_start) & (df_fore["date"] <= period_end)
        else:
            mask = df_fore["date"] == selected_line_date

        line_df = df_fore.loc[mask].copy().sort_values("Timestamp")

        if not line_df.empty:
            y_max = max(line_df["Forecast_Chlorophyll_Kalman"].max(), 10)

            x = line_df["Timestamp"]
            y = line_df["Forecast_Chlorophyll_Kalman"]

            y_good = y.where(y < 4)
            y_warn = y.where((y >= 4) & (y < 8))
            y_danger = y.where(y >= 8)

            fig = go.Figure()
            add_risk_bands_plotly(fig, y_max)

            fig.add_trace(go.Scatter(
                x=x, y=y_good, mode="lines",
                name="좋음 구간",
                line=dict(width=2.0, color="#22c55e"),
                hovertemplate="%{x}<br>클로로필: %{y:.2f} µg/L<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=x, y=y_warn, mode="lines",
                name="주의 구간",
                line=dict(width=2.6, color="#f97316"),
                hovertemplate="%{x}<br>클로로필: %{y:.2f} µg/L<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=x, y=y_danger, mode="lines",
                name="위험 구간",
                line=dict(width=2.8, color="#ef4444"),
                hovertemplate="%{x}<br>클로로필: %{y:.2f} µg/L<extra></extra>",
            ))

            fig.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=45, b=95),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ffffff"),
                xaxis=dict(
                    tickformat="%m-%d %H:%M",
                    gridcolor="rgba(148,163,184,0.25)",
                    zerolinecolor="rgba(148,163,184,0.35)",
                    title="시간",
                    title_font=dict(color="#ffffff", size=12),
                    tickfont=dict(color="#ffffff", size=11),
                ),
                yaxis=dict(
                    range=[0, y_max],
                    gridcolor="rgba(148,163,184,0.25)",
                    zerolinecolor="rgba(148,163,184,0.35)",
                    title="클로로필 (µg/L)",
                    title_font=dict(color="#ffffff", size=12),
                    tickfont=dict(color="#ffffff", size=11),
                ),
                title=dict(
                    text="이번주 시간별 조류 농도 추세",
                    x=0.00, xanchor="left",
                    y=0.95, yanchor="top",
                    font=dict(size=18, color="#ffffff"),
                ),
            )

            if max_info_text:
                fig.add_annotation(
                    x=-0.02, y=-0.50, xref="paper", yref="paper",
                    text=max_info_text,
                    showarrow=False,
                    xanchor="left", yanchor="bottom",
                    align="left",
                    font=dict(size=16, color="#ffffff"),
                )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("선택한 기간에 대한 예측 데이터가 없습니다.")

        # ---------- 7일간 일별 예보 카드 ----------
        week_rows_html = ""
        for _, row in daily.iterrows():
            d = row["date"]

            if today_date is not None and d == today_date:
                day_label = f"오늘 ({d.strftime('%m/%d')})"
            else:
                wd = d.weekday()
                day_label = f"{weekdays_kr[wd]} ({d.strftime('%m/%d')})"

            d_min = row["min"]
            d_max = row["max"]
            d_mean = row["mean"]

            mean_txt = "–" if pd.isna(d_mean) else f"{d_mean:.1f}"
            label, emoji, color, _ = classify_chl(d_mean)

            if denom is None or denom <= 0:
                left_pct = 0
                width_pct = 100
            else:
                left_pct = (float(d_min) - float(global_min)) / float(denom) * 100
                width_pct = (float(d_max) - float(d_min)) / float(denom) * 100
                left_pct = max(0, min(left_pct, 100))
                width_pct = max(5, min(width_pct, 100 - left_pct))

            if denom is None or denom <= 0 or pd.isna(d_mean):
                mean_marker_left = 50.0
            else:
                mean_marker_left = (float(d_mean) - float(global_min)) / float(denom) * 100
                mean_marker_left = max(0, min(mean_marker_left, 100))

            week_rows_html += f"""
  <div class="week-row">
    <div class="week-day">{day_label}</div>
    <div class="week-status">
      <span class="week-emoji">{emoji}</span>
      <span class="week-status-text">{label}</span>
    </div>
    <div class="week-mean">{mean_txt}</div>
    <div class="week-min">{d_min:.1f}</div>
    <div class="week-range-track">
      <div class="week-range-bar"
           style="left:{left_pct:.1f}%; width:{width_pct:.1f}%; background-color:{color};"></div>
      <div class="week-mean-marker"
           style="left:{mean_marker_left:.1f}%;"
           title="평균 {mean_txt} µg/L"></div>
    </div>
    <div class="week-max">{d_max:.1f}</div>
  </div>
"""

        week_card_html = f"""
<div class="card">
  <div class="week-card-header">
    <div class="week-card-title">7일간 일별 예보 (µg/L)</div>
    <div class="week-subtitle">예보 기간: {period_text}</div>
  </div>
  <div class="week-rows">
    <div class="week-header-row">
      <div>요일</div>
      <div>상태</div>
      <div>평균</div>
      <div>최소</div>
      <div>예상 범위</div>
      <div>최대</div>
    </div>
    {week_rows_html}
  </div>
</div>
"""

        map_card_html = """
<div class="card">
  <div class="week-card-header">
    <div class="week-card-title">브리즈번 강 위치</div>
    <div class="week-subtitle">Colmslie Buoy 기준</div>
  </div>
  <div style="position:relative; border-radius: 1.0rem; overflow: hidden; margin-top: 0.25rem;">
    <iframe
        src="https://www.openstreetmap.org/export/embed.html?bbox=153.08047%2C-27.45170%2C153.08647%2C-27.44520&layer=mapnik&marker=-27.44920%2C153.08347"
        style="border:0; width:100%; height:255px;"
        loading="lazy"
        referrerpolicy="no-referrer-when-downgrade">
    </iframe>
    <a
        href="https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=-27.449204719754594,153.0834701552862&heading=0&pitch=0&fov=80"
        target="_blank"
        style="position:absolute; right:0.75rem; bottom:0.75rem; background:rgba(15,23,42,0.85); color:#f9fafb; font-size:0.78rem; padding:0.25rem 0.6rem; border-radius:999px; text-decoration:none;">
        로드뷰 열기
    </a>
  </div>
</div>
"""

        col_week_card, col_map_card = st.columns([3, 2])
        with col_week_card:
            st.markdown(week_card_html, unsafe_allow_html=True)
        with col_map_card:
            st.markdown(map_card_html, unsafe_allow_html=True)

# ============================================================
# 3. 전체 데이터 보기 + 시계열 그래프
# ============================================================
with st.expander("📊 전체 수집 데이터 보기", expanded=False):
    st.markdown(
        """
<div class="expander-text">
- 아래 표는 센서 보정값(Kalman)이 포함된 원시 데이터입니다.<br>
- 원하는 기간과 지표를 선택해 시계열로 볼 수 있고, CSV로 내려받아 추가 분석에 활용할 수 있습니다.
</div>
""",
        unsafe_allow_html=True,
    )

    if not df.empty:
        if "date" in df.columns and "Timestamp" in df.columns:
            min_date = df["date"].min()
            max_date = df["date"].max()

            default_start = max_date - datetime.timedelta(days=2)
            if default_start < min_date:
                default_start = min_date

            start_date, end_date = st.slider(
                "표시 기간 선택",
                min_value=min_date,
                max_value=max_date,
                value=(default_start, max_date),
                format="YYYY-MM-DD",
            )

            mask_range = (df["date"] >= start_date) & (df["date"] <= end_date)
            df_range = df.loc[mask_range].copy()
        else:
            df_range = df.copy()

        numeric_cols = [col for col in df_range.columns if pd.api.types.is_numeric_dtype(df_range[col])]

        if numeric_cols:
            default_idx = numeric_cols.index("Chlorophyll_Kalman") if "Chlorophyll_Kalman" in numeric_cols else 0

            selected_series = st.selectbox(
                "시계열로 보고 싶은 지표",
                options=numeric_cols,
                index=default_idx,
            )

            df_ts = df_range.dropna(subset=["Timestamp"]).sort_values("Timestamp")

            fig_hist = px.line(
                df_ts,
                x="Timestamp",
                y=selected_series,
                labels={"Timestamp": "시간", selected_series: selected_series},
            )
            fig_hist.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=35, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ffffff"),
                xaxis=dict(
                    gridcolor="rgba(148,163,184,0.25)",
                    zerolinecolor="rgba(148,163,184,0.35)",
                    title="시간",
                    title_font=dict(color="#ffffff", size=12),
                    tickfont=dict(color="#ffffff", size=11),
                ),
                yaxis=dict(
                    gridcolor="rgba(148,163,184,0.25)",
                    zerolinecolor="rgba(148,163,184,0.35)",
                    title=selected_series,
                    title_font=dict(color="#ffffff", size=12),
                    tickfont=dict(color="#ffffff", size=11),
                ),
                title=dict(
                    text="선택 지표 시계열",
                    x=0.01,
                    xanchor="left",
                    y=0.95,
                    font=dict(size=14, color="#ffffff"),
                ),
            )

            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("시계열로 표시할 수 있는 수치형 지표가 없습니다.")

        st.dataframe(df_range.tail(300), use_container_width=True)

        csv_all = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 전체 수질 데이터 다운로드 (CSV)",
            data=csv_all,
            file_name="brisbane_water_all.csv",
            mime="text/csv",
        )
    else:
        st.write("데이터가 없습니다.")
