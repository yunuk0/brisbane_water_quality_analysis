import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

# ============================================================
# 기본 설정 & 스타일
# ============================================================
st.set_page_config(
    page_title="브리즈번 수질 모니터링 대시보드",
    page_icon=":droplet:",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .sub-title {
        font-size: 16px;
        opacity: 0.8;
        margin-bottom: 1rem;
    }
    .kpi-card {
        padding: 0.9rem 1.1rem;
        border-radius: 0.9rem;
        background: linear-gradient(135deg, #102a43, #243b53);
        color: white;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.25);
        margin-bottom: 0.8rem;
    }
    .kpi-label {
        font-size: 0.8rem;
        opacity: 0.8;
    }
    .kpi-value {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .kpi-unit {
        font-size: 0.8rem;
        opacity: 0.8;
    }
    .tag-pill {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.7rem;
        margin-right: 0.25rem;
        background-color: #e0f2fe;
        color: #0f172a;
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
        df["month"] = df["Timestamp"].dt.month
        df["day"] = df["Timestamp"].dt.day
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["month"] = pd.to_datetime(df["date"]).dt.month
        df["day"] = pd.to_datetime(df["date"]).dt.day

    return df


@st.cache_data
def load_future_forecast():
    """
    train_offline.py에서 생성한 1주일 예측 파일 로드
    파일 위치: data/future_week_forecast.csv
    """
    path = Path(__file__).parent / "data" / "future_week_forecast.csv"
    df_fore = pd.read_csv(path, parse_dates=["Timestamp"])
    return df_fore


df = get_water_data()

# ============================================================
# 지표 매핑 / Raw vs Kalman 비교 대상 정의
# ============================================================
INDICATOR_MAP = {
    "Chlorophyll_Kalman": "클로로필 (µg/L)",
    "Temperature_Kalman": "수온 (°C)",
    "Dissolved Oxygen_Kalman": "용존산소 (mg/L)",
    "W_Relative Humidity": "산소 포화도 (%)",
    "pH_Kalman": "pH",
    "Salinity_Kalman": "염분 (PSU)",
    "Specific Conductance_Kalman": "전기전도도 (µS/cm)",
    "Turbidity_Kalman": "탁도 (NTU)",
}
AVAILABLE_INDICATORS = [col for col in INDICATOR_MAP.keys() if col in df.columns]

BASE_VARS = [
    "Chlorophyll",
    "Dissolved Oxygen",
    "Salinity",
    "Specific Conductance",
    "Temperature",
]


def add_risk_bands_plotly(fig, y_max):
    """클로로필 농도 위험 구간 배경 + 기준선"""
    fig.add_hrect(y0=0, y1=4,  line_width=0, fillcolor="#d0f0c0", opacity=0.25)
    fig.add_hrect(y0=4, y1=8,  line_width=0, fillcolor="#fff3b0", opacity=0.35)
    fig.add_hrect(y0=8, y1=y_max, line_width=0, fillcolor="#ffc9c9", opacity=0.25)
    fig.add_hline(y=4, line_dash="dash", line_color="orange", line_width=1)
    fig.add_hline(y=8, line_dash="dash", line_color="red",    line_width=1)


# ============================================================
# 사이드바 – 페이지 선택만
# ============================================================
st.sidebar.title("📘 브리즈번 수질 대시보드")

page = st.sidebar.radio(
    "페이지 이동",
    [
        "① 개요",
        "② 추세 분석",
        "③ 지표 비교",
        "④ 기준 초과·예측 경보",
        "⑤ 원시데이터·QA·QC",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("각 페이지별로 분석 기간을 개별 설정할 수 있습니다.")

# ============================================================
# 메인 헤더
# ============================================================
st.markdown(
    '<div class="main-title">🌊 브리즈번 수질 모니터링 & 예측 대시보드</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Brisbane River – Colmslie Water Quality Monitoring Buoy</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <span class="tag-pill">Real-time sensor</span>
    <span class="tag-pill">Water Quality</span>
    <span class="tag-pill">Kalman Filter</span>
    <span class="tag-pill">Risk Monitoring</span>
    """,
    unsafe_allow_html=True,
)

st.write("")

# 공통: date 컬럼 있는지 확인
HAS_DATE = "date" in df.columns

# ============================================================
# ① 개요
# ============================================================
if page == "① 개요":
    st.subheader("① 개요 · 프로젝트 설명 및 핵심 지표 요약")

    # 기본 필터 데이터프레임
    filtered_df = df.copy()
    if HAS_DATE:
        min_date = df["date"].min()
        max_date = df["date"].max()

    col_overview_left, col_overview_right = st.columns([2.2, 1])

    with col_overview_left:
        # ----------------- 프로젝트 설명 -----------------
        with st.expander("프로젝트 개요", expanded=True):
            st.markdown(
                """
                브리즈번 강은 조석, 우기, 도시 유입수의 영향을 동시에 받는 **복합 도시 수역**입니다.  
                이 대시보드는 Colmslie 수질 부이 센서를 활용해 다음을 목표로 합니다.

                - 시간에 따른 수질 패턴 파악  
                - 조류(클로로필)·탁도 등 **오염 리스크 조기 탐지**  
                - Kalman 필터 기반 **센서 노이즈 완화**  
                - 예측 정보를 활용한 **선제적 수질 관리 인사이트 제공**
                """
            )

        # ----------------- KPI 헤더 + 기간 선택 (같은 줄 / 슬라이더) -----------------
        if HAS_DATE:
            kpi_title_col, kpi_date_col = st.columns([1.4, 2.0])
            with kpi_title_col:
                st.markdown("#### 기간 내 주요 지표 평균 (Kalman 처리 기준)")
            with kpi_date_col:
                date_range = st.slider(
                    "분석 기간 선택",
                    min_value=min_date,
                    max_value=max_date,
                    value=(min_date, max_date),
                    key="overview_date_range",
                )

            if isinstance(date_range, tuple):
                start_date, end_date = date_range
            else:
                start_date, end_date = min_date, date_range

            filtered_df = df[
                (df["date"] >= start_date) & (df["date"] <= end_date)
            ].copy()
        else:
            st.markdown("#### 기간 내 주요 지표 평균 (Kalman 처리 기준)")
            st.info("date 컬럼이 없어 전체 기간 기준으로 표시합니다.")

        # ----------------- KPI (분석기간 평균) -----------------
        if not filtered_df.empty:
            avg_values = filtered_df.mean(numeric_only=True)

            k1, k2, k3, k4 = st.columns(4)

            # 🌱 클로로필 평균
            if "Chlorophyll_Kalman" in avg_values.index:
                with k1:
                    st.markdown(
                        f"""
                        <div class="kpi-card">
                          <div class="kpi-label">🌱 클로로필 (평균)</div>
                          <div class="kpi-value">{avg_values['Chlorophyll_Kalman']:.2f}</div>
                          <div class="kpi-unit">µg/L</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # 🌡️ 수온 평균
            if "Temperature_Kalman" in avg_values.index:
                with k2:
                    st.markdown(
                        f"""
                        <div class="kpi-card">
                          <div class="kpi-label">🌡️ 수온 (평균)</div>
                          <div class="kpi-value">{avg_values['Temperature_Kalman']:.2f}</div>
                          <div class="kpi-unit">°C</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # 🫧 용존산소 평균
            if "Dissolved Oxygen_Kalman" in avg_values.index:
                with k3:
                    st.markdown(
                        f"""
                        <div class="kpi-card">
                          <div class="kpi-label">🫧 용존산소 (평균)</div>
                          <div class="kpi-value">{avg_values['Dissolved Oxygen_Kalman']:.2f}</div>
                          <div class="kpi-unit">mg/L</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # 🌫️ 탁도 평균
            if "Turbidity_Kalman" in avg_values.index:
                with k4:
                    st.markdown(
                        f"""
                        <div class="kpi-card">
                          <div class="kpi-label">🌫️ 탁도 (평균)</div>
                          <div class="kpi-value">{avg_values['Turbidity_Kalman']:.2f}</div>
                          <div class="kpi-unit">NTU</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # ----------------- 일별 평균 + 1주일 예측 (+위험 구간) -----------------
            if "Chlorophyll_Kalman" in filtered_df.columns and "date" in filtered_df.columns:
                st.markdown("#### 기간 내 일별 평균 클로로필(조류) 패턴 (+ 1주일 예측)")

                # 실측 일별 평균
                daily_chl = (
                    filtered_df.groupby("date", as_index=False)["Chlorophyll_Kalman"].mean()
                )
                daily_chl = daily_chl.rename(columns={"Chlorophyll_Kalman": "value"})
                daily_chl["series"] = "실측(일별 평균)"

                plot_df = daily_chl.copy()
                forecast_added = False

                # 1주일 예측 파일에서 일별 평균 추출
                try:
                    future_df = load_future_forecast()
                    if (
                        not future_df.empty
                        and "Timestamp" in future_df.columns
                        and "Forecast_Chlorophyll_Kalman" in future_df.columns
                    ):
                        future_daily = future_df.copy()
                        future_daily["date"] = future_daily["Timestamp"].dt.date
                        future_daily = (
                            future_daily.groupby("date", as_index=False)["Forecast_Chlorophyll_Kalman"]
                            .mean()
                        )
                        future_daily = future_daily.rename(
                            columns={"Forecast_Chlorophyll_Kalman": "value"}
                        )
                        future_daily["series"] = "예측(1주일 일별 평균)"

                        plot_df = pd.concat([plot_df, future_daily], ignore_index=True)
                        forecast_added = True
                except FileNotFoundError:
                    forecast_added = False

                fig = px.line(
                    plot_df,
                    x="date",
                    y="value",
                    color="series",
                    labels={
                        "date": "날짜",
                        "value": "클로로필 (µg/L)",
                        "series": "구분",
                    },
                )

                y_max = max(plot_df["value"].max(), 10)
                add_risk_bands_plotly(fig, y_max)

                st.plotly_chart(fig, use_container_width=True)

                if not forecast_added:
                    st.caption("※ 1주일 예측 파일(future_week_forecast.csv)이 없어 실측 데이터만 표시되었습니다.")
        else:
            st.info("선택한 기간에 해당 데이터가 없습니다.")

    with col_overview_right:
        with st.expander("데이터 수집 위치", expanded=True):
            st.markdown("**🗺️ COLMSLIE BOAT RAMP – Sensor Location**")
            brisbane_map = pd.DataFrame(
                {"lat": [-27.449101239198], "lon": [153.08324661695]}
            )
            st.map(brisbane_map)

        with st.expander("데이터셋 개요", expanded=True):
            st.markdown(
                """
                - 출처: Queensland Government Open Data  
                - 지점: Brisbane River – Colmslie Buoy  
                - 수집 간격: 약 10분  
                - 센서 처리: Kalman 필터 적용 파생 컬럼 사용
                """
            )

        with st.expander("🔮 다음 달 예측 (단순 추세)", expanded=False):
            pred_col_display = st.selectbox(
                "예측 변수 선택",
                [
                    "Chlorophyll",
                    "Temperature",
                    "Dissolved Oxygen",
                    "pH",
                    "Salinity",
                    "Specific Conductance",
                    "Turbidity",
                ],
            )
            pred_base_col = f"{pred_col_display}_Kalman"

            if pred_base_col in df.columns:
                if HAS_DATE and not filtered_df.empty and "month" in filtered_df.columns:
                    trend = filtered_df.groupby("month")[pred_base_col].mean().dropna()
                else:
                    trend = df.groupby("month")[pred_base_col].mean().dropna()

                if not trend.empty:
                    months = trend.index.to_numpy(dtype=float)
                    values = trend.values.astype(float)

                    if len(months) > 1:
                        coef = np.polyfit(months, values, 1)
                        last_month = int(months.max())
                        next_month = 1 if last_month == 12 else last_month + 1
                        prediction = coef[0] * next_month + coef[1]
                    else:
                        last_month = int(months[0])
                        next_month = 1 if last_month == 12 else last_month + 1
                        prediction = float(values[0])

                    st.write(
                        f"👉 **{next_month}월 예상 {pred_col_display}: {prediction:.2f}**"
                    )
                else:
                    st.info("선택한 기간에 해당 지표 데이터가 없습니다.")
            else:
                st.warning(f"`{pred_base_col}` 컬럼이 존재하지 않습니다.")

# ============================================================
# ② 추세 분석
# ============================================================
elif page == "② 추세 분석":
    st.subheader("② 추세 분석 · 수질 지표 시간 추세")

    filtered_df = df.copy()
    indicator = AVAILABLE_INDICATORS[0] if AVAILABLE_INDICATORS else None

    if HAS_DATE:
        min_date = df["date"].min()
        max_date = df["date"].max()

        # 시계열 제목 + 기간 선택(슬라이더) + 지표 선택 한 줄 배치
        title_col, ind_col, date_col = st.columns([1.4, 2.0, 2.0])
        with title_col:
            st.markdown("#### 시계열 추세 (라인 차트)")
        with date_col:
            date_range = st.slider(
                "분석 기간 선택",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                key="trend_date_range",
            )
        with ind_col:
            indicator = st.selectbox(
                "추세를 확인할 수질 지표 선택",
                options=AVAILABLE_INDICATORS,
                format_func=lambda x: INDICATOR_MAP.get(x, x),
                key="trend_indicator",
            )

        if isinstance(date_range, tuple):
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, date_range

        filtered_df = df[
            (df["date"] >= start_date) & (df["date"] <= end_date)
        ].copy()
    else:
        st.markdown("#### 시계열 추세 (라인 차트)")
        if AVAILABLE_INDICATORS:
            indicator = st.selectbox(
                "추세를 확인할 수질 지표 선택",
                options=AVAILABLE_INDICATORS,
                format_func=lambda x: INDICATOR_MAP.get(x, x),
            )
        else:
            indicator = None
        st.info("date 컬럼이 없어 전체 기간 기준으로 표시합니다.")

    if filtered_df.empty:
        st.info("선택한 기간에 데이터가 없습니다.")
    elif indicator is None:
        st.info("표시할 수질 지표가 없습니다.")
    else:
        x_col = "Timestamp" if "Timestamp" in filtered_df.columns else "date"

        fig_ts = px.line(
            filtered_df,
            x=x_col,
            y=indicator,
            labels={x_col: "시간", indicator: INDICATOR_MAP.get(indicator, indicator)},
            title=f"{INDICATOR_MAP.get(indicator, indicator)} 시간별 추세",
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        agg_type = st.radio(
            "집계 단위 선택",
            ["월별 평균", "일별 평균"],
            horizontal=True,
        )

        if agg_type == "월별 평균" and "month" in filtered_df.columns:
            monthly = (
                filtered_df.groupby("month", as_index=False)[indicator].mean().dropna()
            )
            if not monthly.empty:
                fig_month = px.bar(
                    monthly,
                    x="month",
                    y=indicator,
                    labels={"month": "월", indicator: INDICATOR_MAP.get(indicator, indicator)},
                    title=f"{INDICATOR_MAP.get(indicator, indicator)} 월별 평균",
                )
                st.plotly_chart(fig_month, use_container_width=True)
        elif agg_type == "일별 평균" and "date" in filtered_df.columns:
            daily = (
                filtered_df.groupby("date", as_index=False)[indicator].mean().dropna()
            )
            if not daily.empty:
                fig_day = px.bar(
                    daily,
                    x="date",
                    y=indicator,
                    labels={"date": "날짜", indicator: INDICATOR_MAP.get(indicator, indicator)},
                    title=f"{INDICATOR_MAP.get(indicator, indicator)} 일별 평균",
                )
                st.plotly_chart(fig_day, use_container_width=True)

        if "predicted_chlorophyll" in filtered_df.columns and "date" in filtered_df.columns:
            st.markdown("#### 예측 조류(클로로필) 추세")
            fig_pred = px.line(
                filtered_df,
                x="date",
                y="predicted_chlorophyll",
                labels={"date": "날짜", "predicted_chlorophyll": "예측 클로로필"},
                title="예측 조류량(클로로필) 추세",
            )
            st.plotly_chart(fig_pred, use_container_width=True)

# ============================================================
# ③ 지표 비교
# ============================================================
elif page == "③ 지표 비교":
    # ---- 분석 기간 / 비교 지표를 타이틀 옆에 배치 (슬라이더) ----
    filtered_df = df.copy()
    compare_cols = []

    if HAS_DATE:
        min_date = df["date"].min()
        max_date = df["date"].max()

        title_col, sel_col, date_col = st.columns([1.8, 2.0, 2.8])
        with title_col:
            st.subheader("③ 지표 비교 · 월별 수질 지표 비교 분석")
        with date_col:
            date_range = st.slider(
                "분석 기간 선택",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                key="compare_date_range",
            )
        with sel_col:
            compare_cols = st.multiselect(
                "비교할 수질 지표 선택 (최대 4개 권장)",
                options=AVAILABLE_INDICATORS,
                default=[
                    c
                    for c in AVAILABLE_INDICATORS
                    if c
                    in [
                        "Chlorophyll_Kalman",
                        "Temperature_Kalman",
                        "Dissolved Oxygen_Kalman",
                    ]
                ][:3],
                format_func=lambda x: INDICATOR_MAP.get(x, x),
                key="compare_cols_multiselect",
                help="Kalman 처리된 수질 지표를 기준으로 월별 평균 및 상관관계를 비교합니다.",
            )

        if isinstance(date_range, tuple):
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, date_range

        filtered_df = df[
            (df["date"] >= start_date) & (df["date"] <= end_date)
        ].copy()
    else:
        st.subheader("③ 지표 비교 · 월별 수질 지표 비교 분석")
        if AVAILABLE_INDICATORS:
            compare_cols = st.multiselect(
                "비교할 수질 지표 선택 (최대 4개 권장)",
                options=AVAILABLE_INDICATORS,
                default=[
                    c
                    for c in AVAILABLE_INDICATORS
                    if c
                    in [
                        "Chlorophyll_Kalman",
                        "Temperature_Kalman",
                        "Dissolved Oxygen_Kalman",
                    ]
                ][:3],
                format_func=lambda x: INDICATOR_MAP.get(x, x),
            )
        st.info("date 컬럼이 없어 전체 기간 기준으로 표시합니다.")

    if filtered_df.empty or "month" not in filtered_df.columns:
        st.info("선택한 기간/데이터로 비교 분석이 어렵습니다.")
    else:
        normalize = st.checkbox("지표 간 스케일 표준화 (z-score)", value=False)

        if compare_cols:
            comp_df = filtered_df[["month"] + compare_cols].copy()

            if normalize:
                for col in compare_cols:
                    m = comp_df[col].mean()
                    s = comp_df[col].std()
                    if s and not np.isnan(s):
                        comp_df[col] = (comp_df[col] - m) / s

            monthly_mean = (
                comp_df.groupby("month")[compare_cols].mean().reset_index().melt(
                    id_vars="month", var_name="indicator", value_name="value"
                )
            ).dropna()

            if not monthly_mean.empty:
                monthly_mean["indicator_label"] = monthly_mean["indicator"].map(
                    INDICATOR_MAP
                )

                fig_cmp = px.bar(
                    monthly_mean,
                    x="month",
                    y="value",
                    color="indicator_label",
                    barmode="group",
                    labels={"month": "월", "value": "값", "indicator_label": "지표"},
                    title="월별 수질 지표 비교 (평균)",
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

            st.markdown("#### 선택 지표 간 상관관계")
            corr_df = filtered_df[compare_cols].corr().round(2)
            fig_corr = px.imshow(
                corr_df,
                text_auto=True,
                aspect="auto",
                title="지표 간 상관계수",
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("비교할 지표를 하나 이상 선택해 주세요.")

# ============================================================
# ④ 기준 초과·예측 경보
# ============================================================
elif page == "④ 기준 초과·예측 경보":
    st.subheader("④ 기준 초과 및 예측 기반 경보 모니터링")

    # --------------------------------------------------------
    # 1) LightGBM 1주일 예측 결과 (가장 위)
    # --------------------------------------------------------
    st.markdown("### 🔬 LightGBM 1주일 예측 결과 (사전 계산 값 사용)")

    try:
        future_df = load_future_forecast()
    except FileNotFoundError:
        st.error("⚠️ 1주일 예측값 파일(future_week_forecast.csv)을 찾을 수 없습니다. 먼저 train_offline.py를 실행해 주세요.")
    else:
        if "Timestamp" in df.columns and "Chlorophyll_Kalman" in df.columns:
            last_real_time = df["Timestamp"].max()
            tail_start = last_real_time - pd.Timedelta(days=7)

            real_tail = df[df["Timestamp"] >= tail_start][["Timestamp", "Chlorophyll_Kalman"]].copy()
            real_tail["series"] = "Kalman 실측 (최근 7일)"
            real_tail = real_tail.rename(columns={"Chlorophyll_Kalman": "value",
                                                  "Timestamp": "time"})

            pred = future_df.copy()
            pred["series"] = "LightGBM 예측 (1주일)"
            pred = pred.rename(columns={"Forecast_Chlorophyll_Kalman": "value"})

            if "Timestamp" in pred.columns:
                pred = pred.rename(columns={"Timestamp": "time"})

            plot_df = pd.concat([real_tail, pred], ignore_index=True)

            fig_future = px.line(
                plot_df,
                x="time",
                y="value",
                color="series",
                labels={"time": "시간", "value": "Chlorophyll (µg/L)", "series": "구분"},
                title="최근 7일 실측 + 1주일 예측 (LightGBM 사전 계산값)",
            )

            y_max = max(plot_df["value"].max(), 10)
            add_risk_bands_plotly(fig_future, y_max)
            fig_future.update_layout(legend_title_text="")
            st.plotly_chart(fig_future, use_container_width=True)

            # 예측값 요약 KPI
            if "Forecast_Chlorophyll_Kalman" in future_df.columns:
                vals = future_df["Forecast_Chlorophyll_Kalman"].dropna()
                if not vals.empty:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("예측 평균", f"{vals.mean():.2f} µg/L")
                    c2.metric("예측 최대", f"{vals.max():.2f} µg/L")
                    c3.metric("예측 최소", f"{vals.min():.2f} µg/L")

            # CSV 다운로드
            csv_data = future_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 1주일 예측값 CSV 다운로드",
                data=csv_data,
                file_name="future_week_forecast.csv",
                mime="text/csv",
            )
        else:
            st.warning("df에 'Timestamp' 또는 'Chlorophyll_Kalman' 컬럼이 없어 예측 결과를 시각화할 수 없습니다.")

    st.markdown("---")

    # --------------------------------------------------------
    # 2) 기간 내 기준 초과 경보 발생 현황 + 일별 요약
    # --------------------------------------------------------
    st.markdown("### 📊 기간 내 기준 초과 경보 발생 현황")

    # ---- 분석 기간 선택 (이 페이지 전용) ----
    if HAS_DATE:
        min_date = df["date"].min()
        max_date = df["date"].max()
        date_range = st.date_input(
            "경보 분석 기간 선택",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="alert_date_range",
        )

        if isinstance(date_range, tuple):
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, date_range

        filtered_df = df[
            (df["date"] >= start_date) & (df["date"] <= end_date)
        ].copy()
    else:
        filtered_df = df.copy()
        st.info("date 컬럼이 없어 전체 기간 기준으로 경보를 분석합니다.")

    if filtered_df.empty or "date" not in filtered_df.columns:
        st.info("선택한 기간에 대한 경보 분석 데이터가 없습니다.")
    else:
        THRESHOLDS = {
            "Chlorophyll_Kalman": {
                "warning": 4,
                "danger": 8,
                "unit": "µg/L",
                "label": "클로로필",
            },
            "Turbidity_Kalman": {
                "warning": 20,
                "danger": 40,
                "unit": "NTU",
                "label": "탁도",
            },
        }
        target_cols = [c for c in THRESHOLDS.keys() if c in filtered_df.columns]

        if not target_cols:
            st.info("기준값이 설정된 지표(클로로필, 탁도)가 데이터에 없습니다.")
        else:
            daily_mean = filtered_df.groupby("date")[target_cols].mean().reset_index()

            alerts = []
            for _, row in daily_mean.iterrows():
                for col in target_cols:
                    val = row[col]
                    if pd.isna(val):
                        continue
                    cfg = THRESHOLDS[col]
                    if val >= cfg["danger"]:
                        level = "위험"
                    elif val >= cfg["warning"]:
                        level = "주의"
                    else:
                        continue
                    alerts.append(
                        {
                            "date": row["date"],
                            "지표": cfg["label"],
                            "평균값": round(val, 2),
                            "수준": level,
                            "단위": cfg["unit"],
                        }
                    )

            if alerts:
                alert_df = pd.DataFrame(alerts)

                # 1) 기간 내 기준 초과 경보 발생 현황 (그래프 우선)
                daily_alert_cnt = (
                    alert_df.groupby(["date", "수준"])
                    .size()
                    .reset_index(name="건수")
                )
                fig_alert = px.bar(
                    daily_alert_cnt,
                    x="date",
                    y="건수",
                    color="수준",
                    barmode="stack",
                    title="기간 내 기준 초과 경보 발생 현황",
                    labels={"date": "날짜", "건수": "경보 건수"},
                )
                st.plotly_chart(fig_alert, use_container_width=True)

                # 2) 기준 초과 일별 요약 (테이블)
                st.markdown("#### 기준 초과 일별 요약")
                st.dataframe(alert_df, use_container_width=True, hide_index=True)
            else:
                st.success("선택한 기간 동안 정의된 기준값을 초과한 일별 평균은 없습니다.")

# ============================================================
# ⑤ 원시데이터·QA·QC
# ============================================================
elif page == "⑤ 원시데이터·QA·QC":
    st.subheader("⑤ 원시데이터 · QA·QC (Raw vs Kalman 비교)")

    # ---- 분석 기간 선택 (이 페이지 전용) ----
    if HAS_DATE:
        min_date = df["date"].min()
        max_date = df["date"].max()
        date_range = st.date_input(
            "분석 기간 선택",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="qa_date_range",
        )

        if isinstance(date_range, tuple):
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, date_range

        filtered_df = df[
            (df["date"] >= start_date) & (df["date"] <= end_date)
        ].copy()
    else:
        filtered_df = df.copy()
        st.info("date 컬럼이 없어 전체 기간 기준으로 표시합니다.")

    if filtered_df.empty:
        st.info("선택한 기간에 해당하는 데이터가 없습니다.")
    else:
        total_rows = len(filtered_df)
        n_days = filtered_df["date"].nunique() if "date" in filtered_df.columns else None

        c1, c2 = st.columns(2)
        c1.metric("레코드 수", f"{total_rows:,}")
        if n_days is not None:
            c2.metric("관측 일수", f"{n_days}일")

        st.markdown("---")

        time_cols = [c for c in ["Timestamp", "date"] if c in filtered_df.columns]

        st.markdown("### 1. 원시 수질 데이터 품질 요약 (Raw)")

        raw_cols_in_df = [b for b in BASE_VARS if b in filtered_df.columns]

        if raw_cols_in_df:
            raw_missing = (
                filtered_df[raw_cols_in_df]
                .isna()
                .sum()
                .reset_index()
            )
            raw_missing.columns = ["컬럼", "결측치 개수"]
            raw_missing["결측률(%)"] = (
                raw_missing["결측치 개수"] / total_rows * 100
            ).round(2)

            st.markdown("#### ▪ 결측치 현황 (Raw)")
            st.dataframe(raw_missing, use_container_width=True, hide_index=True)

            raw_stats = (
                filtered_df[raw_cols_in_df]
                .describe()
                .T[["mean", "std", "min", "max"]]
                .round(3)
                .reset_index()
            )
            raw_stats.columns = ["컬럼", "평균", "표준편차", "최소값", "최대값"]

            st.markdown("#### ▪ 기본 통계 (Raw)")
            st.dataframe(raw_stats, use_container_width=True, hide_index=True)
        else:
            st.info("원시 수질 컬럼(Chlorophyll, Temperature 등)이 존재하지 않습니다.")

        st.markdown("---")

        st.markdown("### 2. Kalman 처리 데이터 품질 요약")

        kalman_cols_in_df = [
            f"{b}_Kalman" for b in BASE_VARS if f"{b}_Kalman" in filtered_df.columns
        ]

        if kalman_cols_in_df:
            kal_missing = (
                filtered_df[kalman_cols_in_df]
                .isna()
                .sum()
                .reset_index()
            )
            kal_missing.columns = ["컬럼", "결측치 개수"]
            kal_missing["결측률(%)"] = (
                kal_missing["결측치 개수"] / total_rows * 100
            ).round(2)

            st.markdown("#### ▪ 결측치 현황 (Kalman)")
            st.dataframe(kal_missing, use_container_width=True, hide_index=True)

            kal_stats = (
                filtered_df[kalman_cols_in_df]
                .describe()
                .T[["mean", "std", "min", "max"]]
                .round(3)
                .reset_index()
            )
            kal_stats.columns = ["컬럼", "평균", "표준편차", "최소값", "최대값"]

            st.markdown("#### ▪ 기본 통계 (Kalman)")
            st.dataframe(kal_stats, use_container_width=True, hide_index=True)
        else:
            st.info("Kalman 처리 컬럼(*_Kalman)이 존재하지 않습니다.")

        st.markdown("---")

        st.markdown("### 3. Kalman 처리 효과 비교 (Raw vs Kalman)")

        comparison_rows = []
        for base in BASE_VARS:
            raw_col = base
            kalman_col = f"{base}_Kalman"

            if raw_col in filtered_df.columns and kalman_col in filtered_df.columns:
                raw_series = filtered_df[raw_col]
                kal_series = filtered_df[kalman_col]

                if raw_series.notna().sum() == 0 or kal_series.notna().sum() == 0:
                    continue

                raw_mean = float(raw_series.mean())
                kal_mean = float(kal_series.mean())
                raw_std = float(raw_series.std())
                kal_std = float(kal_series.std())

                if raw_std > 0:
                    reduction = (raw_std - kal_std) / raw_std * 100
                else:
                    reduction = np.nan

                comparison_rows.append(
                    {
                        "지표": base,
                        "Raw 평균": round(raw_mean, 3),
                        "Kalman 평균": round(kal_mean, 3),
                        "Raw 표준편차": round(raw_std, 3),
                        "Kalman 표준편차": round(kal_std, 3),
                        "표준편차 감소율(%)": round(reduction, 1)
                        if not np.isnan(reduction)
                        else np.nan,
                    }
                )

        if comparison_rows:
            comp_df = pd.DataFrame(comparison_rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        else:
            st.info("Raw 컬럼과 Kalman 컬럼이 동시에 존재하는 지표가 없습니다.")

        st.markdown("---")

        st.markdown("### 4. 데이터 샘플 (Raw / Kalman)")

        col_raw_sample, col_kal_sample = st.columns(2)

        with col_raw_sample:
            st.markdown("#### ▪ Raw 수질 데이터 샘플 (상위 200행)")
            if raw_cols_in_df:
                st.dataframe(
                    filtered_df[time_cols + raw_cols_in_df].head(200),
                    use_container_width=True,
                )
            else:
                st.write("표시할 Raw 수질 컬럼이 없습니다.")

        with col_kal_sample:
            st.markdown("#### ▪ Kalman 수질 데이터 샘플 (상위 200행)")
            if kalman_cols_in_df:
                st.dataframe(
                    filtered_df[time_cols + kalman_cols_in_df].head(200),
                    use_container_width=True,
                )
            else:
                st.write("표시할 Kalman 수질 컬럼이 없습니다.")

        st.markdown("---")

        csv_data = filtered_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 필터 적용 전체 데이터 CSV 다운로드",
            data=csv_data,
            file_name="brisbane_water_filtered.csv",
            mime="text/csv",
        )
