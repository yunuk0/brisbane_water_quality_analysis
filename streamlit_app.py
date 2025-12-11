import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import datetime

# ============================================================
# 기본 설정 & 스타일 (날씨앱 느낌 + 반응형)
# ============================================================
st.set_page_config(
    page_title="브리즈번 수질 알리미",
    page_icon=":droplet:",
    layout="wide",
)

st.markdown(
    """
<style>
.stApp {
    background: radial-gradient(circle at top, #0b1020 0, #020617 55%, #020617 100%);
    color: #e5e7eb;
}

/* 기본 padding: 모바일 기준 */
.block-container {
    padding-top: 3.5rem;
    padding-bottom: 2rem;
    padding-left: 1.2rem;
    padding-right: 1.2rem;
}

/* 큰 화면에서만 좌우 여유를 더 줌 */
@media (min-width: 1200px) {
  .block-container {
      padding-left: 5rem;
      padding-right: 5rem;
  }
}

.main-title {
    font-size: clamp(22px, 2.3vw, 30px);
    font-weight: 800;
    margin-bottom: 0.25rem;
    color: #f9fafb;
}
.sub-title {
    font-size: 14px;
    opacity: 0.8;
    margin-bottom: 1rem;
}
.tag-pill {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    font-size: 0.7rem;
    margin-right: 0.25rem;
    background-color: rgba(15, 23, 42, 0.8);
    color: #e5e7eb;
    border: 1px solid rgba(148, 163, 184, 0.4);
}

/* =========================================================
   메인 카드 (모바일 = 1단, 데스크톱 = 2단)
   ========================================================= */
.hero-card {
    padding: 1.2rem 1.4rem;
    border-radius: 1.3rem;
    background: radial-gradient(circle at top, #1d2752, #020617);
    color: #e5e7eb;
    box-shadow: 0 20px 40px rgba(15, 60, 42, 0.9);

    /* 모바일 기본: 세로 1단 */
    display: grid;
    grid-template-columns: 1fr;
    row-gap: 1.2rem;

    min-height: 260px;
    height: auto;
}

/* 데스크톱 이상에서만 좌/우 2열 */
@media (min-width: 900px) {
  .hero-card {
      grid-template-columns: 2fr 1.1fr;
      column-gap: 2rem;
  }
}

/* 왼쪽 열: 텍스트 블록 세로 중앙 정렬 */
.hero-left {
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.hero-title {
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.7;
}
.hero-location {
    font-size: 1.1rem;
    margin-top: 0.2rem;
    font-weight: 600;
}

/* 수치 한 줄 */
.hero-main-row {
    display: flex;
    align-items: flex-end;
    flex-wrap: wrap;           /* 좁은 화면에서 줄바꿈 허용 */
    gap: 0.2rem;
    margin-top: 0.5rem;
}
.hero-main-value {
    font-size: clamp(2.4rem, 6vw, 3.5rem);
    font-weight: 800;
    line-height: 1.05;
}
.hero-main-unit {
    font-size: 1.1rem;
    opacity: 0.8;
    margin-bottom: 0.3rem;
}

.hero-label {
    font-size: 0.85rem;
    opacity: 0.75;
    margin-top: 0.4rem;
    margin-bottom: 0.05rem;
}
.hero-subtext {
    font-size: 0.78rem;
    opacity: 0.8;
    margin-top: 0rem;
}
.hero-subtext-note {
    font-size: 0.75rem;
    opacity: 0.6;
    margin-top: 0.2rem;
}

/* 오른쪽 열: 상태 박스 */
.hero-status-box {
    display: flex;
    align-items: center;
    justify-content: center;
}

/* 상태 배지 (좋음/주의/위험) – 반응형 캡슐 박스 */
.hero-badge {
    width: 100%;
    max-width: 420px;
    height: 100%;
    max-height: 180px;

    padding: 0 1.6rem;
    border-radius: 999px;
    background-color: rgba(15, 23, 42, 0.9);
    border: 2px solid rgba(148, 163, 184, 0.5);
    box-sizing: border-box;

    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
}
.hero-badge span:first-child {
    font-size: 1.4rem;
}
.hero-badge-label {
    font-size: clamp(2.2rem, 5vw, 4rem);
    font-weight: 700;
}

/* 모바일에서 배지 최소 높이 확보 */
@media (max-width: 899px) {
  .hero-status-box {
      min-height: 140px;
  }
}

/* chip 카드들 */
.chip-box {
    padding: 0.75rem 0.9rem;
    border-radius: 1rem;
    background-color: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.35);
    font-size: 0.78rem;
    margin-bottom: 0.4rem;
}
.chip-label {
    opacity: 0.7;
    font-size: 0.76rem;
}
.chip-value {
    font-size: 1.05rem;
    font-weight: 600;
    margin-top: 0.2rem;
}
.small-title {
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 0.25rem;
    margin-top: 0.8rem;
}
.section-title {
    font-size: 1.0rem;
    font-weight: 700;
    margin-top: 1.2rem;
    margin-bottom: 0.5rem;
}
.info-text {
    font-size: 0.8rem;
    opacity: 0.75;
}

/* 아주 작은 화면에서 표, 차트 여백 살짝 조정 */
@media (max-width: 600px) {
  .section-title {
      margin-top: 1rem;
  }
}

/* ===== Metric 색상 커스터마이즈 (예보 평균/최대 등) ===== */
div[data-testid="stMetricLabel"] {
    color: #f9fafb !important;   /* 라벨: 예보 평균, 예보 최대, 위험 구간 등 */
}
div[data-testid="stMetricValue"] {
    color: #f9fafb !important;   /* 값 숫자 */
}
div[data-testid="stMetricDelta"] {
    color: #f97316 !important;   /* 증감(▲/▼) 사용하는 경우 */
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# 데이터 로드
# ============================================================
@st.cache_data
def get_water_data():
    DATA_FILENAME = Path(__file__).parent / "data" / "df_final.csv"
    df = pd.read_csv(DATA_FILENAME)

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["date"] = df["Timestamp"].dt.date
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    return df


@st.cache_data
def load_future_forecast():
    """
    1주일 예측 CSV (10분 단위) 로드
    컬럼: Timestamp, Forecast_Chlorophyll_Kalman
    """
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


def get_last_valid(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return np.nan
    return df[col].dropna().iloc[-1] if df[col].notna().any() else np.nan


def add_risk_bands_plotly(fig, y_max):
    fig.add_hrect(y0=0, y1=4,  line_width=0, fillcolor="#d0f0c0", opacity=0.25)
    fig.add_hrect(y0=4, y1=8,  line_width=0, fillcolor="#fff3b0", opacity=0.35)
    fig.add_hrect(y0=8, y1=y_max, line_width=0, fillcolor="#ffc9c9", opacity=0.25)
    fig.add_hline(y=4, line_dash="dash", line_color="orange", line_width=1)
    fig.add_hline(y=8, line_dash="dash", line_color="red",    line_width=1)

# ============================================================
# 기본 정보 계산
# ============================================================
if "Timestamp" in df.columns:
    df = df.sort_values("Timestamp")
    latest_row = df.iloc[-1]
    latest_time = latest_row["Timestamp"]
    today_date = latest_time.date()

    last_24h_df = df[df["Timestamp"] >= latest_time - pd.Timedelta(hours=24)].copy()
else:
    latest_row = df.iloc[-1]
    latest_time = None
    today_date = df["date"].iloc[-1] if "date" in df.columns else None
    last_24h_df = df.copy()

cur_chl = get_last_valid(df, "Chlorophyll_Kalman")
cur_temp = get_last_valid(df, "Temperature_Kalman")
cur_do = get_last_valid(df, "Dissolved Oxygen_Kalman")
cur_turb = get_last_valid(df, "Turbidity_Kalman")

level_label, level_emoji, level_color, level_msg = classify_chl(cur_chl)

# 오늘 최소·최대 조류
if "date" in df.columns and today_date is not None:
    today_df = df[df["date"] == today_date]
else:
    today_df = last_24h_df

if "Chlorophyll_Kalman" in today_df.columns and not today_df["Chlorophyll_Kalman"].dropna().empty:
    today_min = today_df["Chlorophyll_Kalman"].min()
    today_max = today_df["Chlorophyll_Kalman"].max()
else:
    today_min = np.nan
    today_max = np.nan

# 7일 예측 중 최대값 시점
max_future_value = None
max_future_time = None
if forecast_df is not None and not forecast_df.empty:
    idxmax = forecast_df["Forecast_Chlorophyll_Kalman"].idxmax()
    max_future_value = forecast_df.loc[idxmax, "Forecast_Chlorophyll_Kalman"]
    max_future_time = forecast_df.loc[idxmax, "Timestamp"]

# ============================================================
# 헤더
# ============================================================
st.markdown(
    '<div class="main-title">브리즈번 수질 알리미</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">브리즈번 강(Colmslie Buoy) 수질을 날씨앱처럼 쉽게 확인하세요.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
<span class="tag-pill">실시간 센서</span>
<span class="tag-pill">조류(녹조) 모니터링</span>
<span class="tag-pill">7일 예보</span>
<span class="tag-pill">市민용 안내</span>
""",
    unsafe_allow_html=True,
)
st.write("")

# ============================================================
# 1. 오늘의 브리즈번 강 상태
# ============================================================
col_hero_main, col_hero_side = st.columns([2, 1.4])

with col_hero_main:
    chl_text = "–" if pd.isna(cur_chl) else f"{cur_chl:.1f}"

    hero_html = f"""<div class="hero-card">
<div class="hero-left">
  <div class="hero-title">TODAY • BRISBANE RIVER</div>
  <div class="hero-location">Colmslie 수질 관측 부이</div>

  <div class="hero-main-row">
    <span class="hero-main-value">{chl_text}</span>
    <span class="hero-main-unit">µg/L</span>
  </div>

  <div class="hero-label">현재 조류(클로로필) 보정값 기준</div>
  <div class="hero-subtext">{level_msg}</div>
  <div class="hero-subtext hero-subtext-note">
    ※ 환경부·호주 환경기준 참고(0–4 µg/L 양호, 4–8 주의, 8 이상 위험)
  </div>
</div>

<div class="hero-status-box">
  <div class="hero-badge" style="border-color:{level_color};">
    <span>{level_emoji}</span>
    <span class="hero-badge-label" style="color:{level_color};">{level_label}</span>
  </div>
</div>
</div>"""

    st.markdown(hero_html, unsafe_allow_html=True)

with col_hero_side:
    st.markdown('<div class="small-title">오늘 조류(녹조) 범위</div>', unsafe_allow_html=True)
    range_text = (
        f"{today_min:.1f} ~ {today_max:.1f} µg/L"
        if not pd.isna(today_min)
        else "데이터 없음"
    )
    st.markdown(
        f"""<div class="chip-box">
<div class="chip-label">오늘 최소 · 최대 (보정값 기준)</div>
<div class="chip-value">{range_text}</div>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="small-title">현재 주요 지표</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        temp_text = "–" if pd.isna(cur_temp) else f"{cur_temp:.1f} °C"
        st.markdown(
            f"""<div class="chip-box">
<div class="chip-label">수온</div>
<div class="chip-value">{temp_text}</div>
</div>""",
            unsafe_allow_html=True,
        )
    with c2:
        turb_text = "–" if pd.isna(cur_turb) else f"{cur_turb:.1f} NTU"
        st.markdown(
            f"""<div class="chip-box">
<div class="chip-label">탁도</div>
<div class="chip-value">{turb_text}</div>
</div>""",
            unsafe_allow_html=True,
        )

    c3, c4 = st.columns(2)
    with c3:
        do_text = "–" if pd.isna(cur_do) else f"{cur_do:.1f} mg/L"
        st.markdown(
            f"""<div class="chip-box">
<div class="chip-label">용존산소</div>
<div class="chip-value">{do_text}</div>
</div>""",
            unsafe_allow_html=True,
        )
    with c4:
        if latest_time is not None:
            time_txt = latest_time.strftime("%Y-%m-%d %H:%M")
        else:
            time_txt = "정보 없음"
        st.markdown(
            f"""<div class="chip-box">
<div class="chip-label">마지막 업데이트 시각</div>
<div class="chip-value">{time_txt}</div>
</div>""",
            unsafe_allow_html=True,
        )

# ============================================================
# 2. 앞으로 7일 조류(녹조) 예보 – 10분 단위 라인 + 애니메이션
# ============================================================
st.markdown('<div class="section-title" style="font-size:1.3rem;">📆 Chlorophyll(조류) 예보[7일]</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="info-text">센서 데이터를 학습한 예측 모델을 이용해, 약 10분 간격으로 예측한 조류 농도(µg/L)를 시간 흐름에 따라 보여줍니다.</div>',
    unsafe_allow_html=True,
)

if forecast_df is None or forecast_df.empty:
    st.info("예측 파일(future_week_forecast.csv)을 찾을 수 없어, 7일 예보를 표시할 수 없습니다.")
else:
    base = forecast_df[["Timestamp", "Forecast_Chlorophyll_Kalman"]].dropna().copy()
    base = base.sort_values("Timestamp").reset_index(drop=True)

    frames = []
    n = len(base)
    for i in range(n):
        tmp = base.iloc[: i + 1].copy()
        tmp["frame"] = i
        frames.append(tmp)
    anim_df = pd.concat(frames, ignore_index=True)

    chl_max_fore = base["Forecast_Chlorophyll_Kalman"].max()
    y_max = chl_max_fore if chl_max_fore >= 10 else 10

    ANIM_SPEED_MS = 1

    fig_fore = px.line(
        anim_df,
        x="Timestamp",
        y="Forecast_Chlorophyll_Kalman",
        animation_frame="frame",
        range_x=[base["Timestamp"].min(), base["Timestamp"].max()],
        range_y=[0, y_max],
        labels={
            "Timestamp": "시간",
            "Forecast_Chlorophyll_Kalman": "예상 클로로필 (µg/L)",
            "frame": "예측 진행",
        },
    )

    add_risk_bands_plotly(fig_fore, y_max)

    fig_fore.update_layout(
        legend_title_text="",
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        # 대시보드 배경과 맞추기
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        font=dict(color="#e5e7eb"),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.15)",
            zerolinecolor="rgba(148,163,184,0.2)",
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.15)",
            zerolinecolor="rgba(148,163,184,0.2)",
        ),
    )

    # 재생/멈춤 버튼 위치 & 속도 조절
    if fig_fore.layout.updatemenus and len(fig_fore.layout.updatemenus) > 0:
        um = fig_fore.layout.updatemenus[0]
        um.x = 0
        um.xanchor = "left"
        um.y = 1.05
        um.yanchor = "bottom"
        um.pad = dict(l=0, r=0, t=0, b=0)
        for btn in um.buttons:
            if "args" in btn and len(btn["args"]) > 1:
                args1 = btn["args"][1]
                if "frame" in args1:
                    args1["frame"]["duration"] = ANIM_SPEED_MS
                if "transition" in args1:
                    args1["transition"]["duration"] = int(ANIM_SPEED_MS / 2)

    # 프레임 슬라이더 라벨 조정
    frame_labels = {
        i: ts.strftime("%Y-%m-%d %H:%M")
        for i, ts in enumerate(base["Timestamp"])
    }
    if fig_fore.layout.sliders and len(fig_fore.layout.sliders) > 0:
        slider = fig_fore.layout.sliders[0]
        slider.x = 0
        slider.xanchor = "left"
        slider.len = 1.0
        slider.pad = dict(l=0, r=0, t=50, b=0)
        for i, step in enumerate(slider["steps"]):
            step["label"] = frame_labels.get(i, step["label"])

    st.plotly_chart(fig_fore, use_container_width=True)

    # 예보 요약 메트릭
    c1, c2, c3 = st.columns(3)
    vals = base["Forecast_Chlorophyll_Kalman"]
    with c1:
        st.metric("예보 평균", f"{vals.mean():.1f} µg/L")
    with c2:
        st.metric("예보 최대", f"{vals.max():.1f} µg/L")
    with c3:
        high_points = (vals >= 8).sum()
        st.metric("위험 구간(≥8) 시점 수", f"{int(high_points)}개")

    if max_future_time is not None:
        lab, emo, _, _ = classify_chl(max_future_value)
        t_txt = max_future_time.strftime("%Y-%m-%d %H:%M")
        st.markdown(
            f"""
<div class="info-text" style="margin-top:0.4rem;">
  가장 조류 농도가 높게 예보된 시점은 <b>{t_txt}</b>이며,  
  예측값은 약 <b>{max_future_value:.1f} µg/L</b> ({emo} {lab}) 입니다.
</div>
""",
            unsafe_allow_html=True,
        )

# ============================================================
# 3. 데이터 자세히 보기
# ============================================================
with st.expander("📊 데이터 자세히 보기 · 그래프·표 (관심자/전문가용)", expanded=False):
    st.markdown(
        """
- 아래 표는 센서 보정값(Kalman)이 포함된 원시 데이터 일부입니다.  
- 엑셀로 내려받아 추가 분석도 가능합니다.
""",
        unsafe_allow_html=True,
    )

    if "date" in df.columns and today_date is not None:
        recent_start = today_date - datetime.timedelta(days=2)
        recent_mask = df["date"] >= recent_start
        df_recent = df[recent_mask].copy()
    else:
        df_recent = df.tail(500).copy()

    st.dataframe(df_recent.tail(300), use_container_width=True)

    csv_all = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 전체 수질 데이터 다운로드 (CSV)",
        data=csv_all,
        file_name="brisbane_water_all.csv",
        mime="text/csv",
    )
