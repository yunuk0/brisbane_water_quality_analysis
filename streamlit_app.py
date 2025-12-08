import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import numpy as np

st.set_page_config(
    page_title='브리즈번 수질 모니터링',
    page_icon=':droplet:')

@st.cache_data
def get_water_data():
    DATA_FILENAME = Path(__file__).parent / 'data/df_final.csv'
    return pd.read_csv(DATA_FILENAME)
    
df = get_water_data()

page = st.sidebar.selectbox('Go to', ['프로젝트 개요', '월별 수질 경향', '수질 지표 예측'])

st.sidebar.title("📘 프로젝트 요약")

with st.sidebar.expander("📌 웹앱 개요", expanded=True):
    st.markdown("""
    브리즈번 강 수질 데이터를 기반으로 **월별 수질 경향**과 
    **지표 예측** 기능을 제공합니다.""")

if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['month'] = df['Timestamp'].dt.month
    df['day'] = df['Timestamp'].dt.day
    df['date'] = df['Timestamp'].dt.date

with st.sidebar.expander("🔮 다음 달 예측", expanded=True):
    pred_col = st.selectbox("예측 변수 선택", ['Chlorophyll', 'Temperature', 'Dissolved Oxygen', 'pH', 'Salinity', 'Specific Conductance', 'Turbidity'])

    trend = df.groupby('month')[f"{pred_col}_Kalman"].mean()

    months = np.array(trend.index)
    values = np.array(trend.values)

    coef = np.polyfit(months, values, 1)
    next_month = 1
    prediction = coef[0] * next_month + coef[1]

    st.write(f"👉 **2026년 {next_month}월 예상 {pred_col} 수치: {prediction:.2f}**")

# -----------------------------------------------------------------------------

if page == '프로젝트 개요':

    st.title("🌊 수질 모니터링 기반 오염 예측")
    st.subheader('''**_Water Quality Monitoring & Predictive Analytics_**''')

    # ------------------------
    # 01. 개요
    # ------------------------
    with st.expander("**프로젝트 개요**", expanded=True):
        st.markdown('''
        브리즈번 강은 도시 재생과 복원사업을 통해 산책로, 레저 활동, 관광 어트랙션 등이 활성화된 브리즈번의 핵심 생활·공간입니다.
        이처럼 강의 이용도가 높기 때문에 수질 상태에 대한 정확한 정보는 시민의 안전과 도시 운영에 매우 중요합니다. 
        호주정부에서는 강의 수질을 주기적으로 모니터링하고, 그 데이터를 실시간 또는 주기적으로 시민들에게 공개함으로써 누구나 강 주변을 안전하게 즐길 수 있도록 관리하고 있습니다.
        하지만 브리즈번 강은 조석의 영향을 강하게 받는 하천으로, 물의 흐름과 수질 특성이 짧은 시간에도 크게 변해 시간 간격이 긴 일반적인 모니터링으로는 빠르게 변화하는 수질과 유속 패턴을 파악하기 어렵습니다.
        
        본 프로젝트는 브리즈번(Brisbane) 지역 주요 하천 및 하구에서 측정된 실측 수질 센서 데이터를 기반으로, 시간에 따라 변화하는 수질 상태를 진단하고 오염 위험을 예측하는 환경 데이터 사이언스 분석 프로젝트입니다.
        이 프로젝트는 단순한 데이터 시각화 수준을 넘어 **도시 수역의 환경 패턴 이해**, **오염 위험 조기 탐지**, **센서 기반 AI 모델의 가능성 탐색**을 핵심 목표로 합니다.
        ''')

    # ------------------------
    # 02. 브리즈번 지역 환경 배경
    # ------------------------
    with st.expander("**브리즈번 지역 환경 배경**", expanded=True):
        st.markdown("#### 🏞 브리즈번 강(Brisbane River)")
        st.write('''
        브리즈번 강은 길이 약 344km의 호주 퀸즐랜드주의 대표적인 하천으로, 도시 중심(Brisbane CBD)을 관통하며 조석·도시 유입·기후 영향이 복합적으로 작용해 수질 변동성이 높습니다.
        ''')

        st.markdown("#### 🌊 도시 수질 관리의 중요성")
        st.markdown('''
        1. **조석(tide) 영향** → 염분·전기전도도·수온 패턴 변동  
        2. **우기(12~3월)** → 탁도 상승, 오염물질 유입 증가  
        3. **도시 배출수·산업 영향** → pH·염분·클로로필 급격한 변화  
        4. **기후 패턴 변화** → 수온·산소포화도 변동  
        ''')

    # ------------------------
    # 03. 데이터셋 상세 정보
    # ------------------------
    with st.expander("**데이터셋 상세 정보**", expanded=True):

        st.markdown("#### 💿 데이터 출처")
        st.markdown("""
            본 대시보드에서 사용하는 수질 데이터는 **퀸즐랜드 정부 오픈 데이터 포털**에서 제공됩니다.  

            🔗 [Brisbane River - Colmslie Site Water Quality Monitoring Buoy 데이터셋](https://www.data.qld.gov.au/dataset/brisbane-river-colmslie-site-water-quality-monitoring-buoy/resource/0ec4dacc-8e78-4c2a-aa70-d7865ec098e2)

            데이터는 실시간 혹은 주기적으로 업데이트되며, 기타 세부 정보(측정 항목, 기록 방식, 라이선스 등)는 제공처인 Queensland Government의 페이지에서 확인할 수 있습니다.
            """)
        st.markdown("#### 📌 데이터 개요")
        st.markdown('''
        - 전체 데이터: **30,894 rows × 20 columns**  
        - 수집 기간: **2023년 12월 5일 ~ 2025년 3월 10일**  
        - 수집 간격: **10분 단위**  
        ''')

        st.markdown("#### 📊 주요 수질 지표 요약")
        st.markdown('''
        - **Chlorophyll (µg/L)**: 조류량  
        - **Turbidity (NTU)**: 탁도  
        - **Temperature (°C)**: 수온  
        - **pH**: 산성/알칼리성  
        - **Salinity (PSU)**: 염분  
        - **Specific Conductance (µS/cm)**: 전기전도도  
        - **Dissolved Oxygen (mg/L)**: 용존산소  
        - **Relative Humidity (%)**: 산소 포화도  
        - **Weather Temperature (°C)**: 기온  
        - **Shortwave Radiation**: 일사량  
        ''')

    # ------------------------
    # 04. 지도
    # ------------------------
    with st.expander("**데이터 수집 위치**", expanded=True):
        st.markdown("#### 🗺️ COLMSLIE BOAT RAMP - Sensor Location")
        brisbane_map = pd.DataFrame({'lat': [-27.449101239198], 'lon': [153.083246616950]})
        st.map(brisbane_map)



# --- 월별 수질 경향 페이지 ---
elif page == '월별 수질 경향':
    st.title('📊 월별 수질 평균 대시보드')

    # 드롭다운 - 월 선택
    selected_month = st.radio(
        '월 선택',
        sorted(df['month'].unique()),
        horizontal=True)

    month_df = df[df['month'] == selected_month]
    avg_values = month_df.mean(numeric_only=True)

    st.subheader(f'📌 {selected_month}월 주요 수질 평균')

    cols = st.columns(4)
    cols[0].metric('평균 클로로필 농도', f"{avg_values['Chlorophyll_Kalman']:.2f}")
    cols[1].metric('평균 수온', f"{avg_values['Temperature_Kalman']:.2f}")
    cols[2].metric('평균 용존산소', f"{avg_values['Dissolved Oxygen_Kalman']:.2f}")
    cols[3].metric('평균 산소 포화도', f"{avg_values['W_Relative Humidity']:.2f}")

    cols2 = st.columns(4)
    cols2[0].metric('평균 pH', f"{avg_values['pH_Kalman']:.2f}")
    cols2[1].metric('평균 염분 농도', f"{avg_values['Salinity_Kalman']:.2f}")
    cols2[2].metric('평균 전기전도도', f"{avg_values['Specific Conductance_Kalman']:.2f}")
    cols2[3].metric('평균 탁도', f"{avg_values['Turbidity_Kalman']:.2f}")

    daily_avg = month_df.groupby('day', as_index=False)['Chlorophyll_Kalman'].mean()

    fig = px.bar(daily_avg, x='day', y='Chlorophyll_Kalman',
                 title=f'🗓️ {selected_month}월 일별 조류량', color_discrete_sequence=["#3E3F40"],
                 labels={'day': '일', 'Chlorophyll_Kalman': '평균 클로로필 농도 (µg/L)'})

    fig.add_hrect(y0=0, y1=4, fillcolor="green", opacity=0.05, line_width=0, layer="below")
    fig.add_hrect(y0=4, y1=8, fillcolor="yellow", opacity=0.05, line_width=0, layer="below")
    fig.add_hrect(y0=8, y1=daily_avg['Chlorophyll_Kalman'].max() + 2,
                fillcolor="red", opacity=0.05, line_width=0, layer="below")

    fig.add_annotation(
        text=(
            "<b>클로로필(조류) 농도 구간 안내</b><br>"
            "<span style='color:green;'>■ 좋음 (0–4 µg/L)</span><br>"
            "<span style='color:orange;'>■ 주의 (4–8 µg/L)</span><br>"
            "<span style='color:red;'>■ 나쁨 (8 µg/L 이상)</span>"
        ),
        xref="paper", yref="paper", x=0, y=-0.5, showarrow=False, align="left")

    fig.update_layout(margin=dict(b=150))

    st.plotly_chart(fig, use_container_width=True)



# --- 수질 지표 예측 페이지 ---
elif page == '수질 지표 예측':
    st.title('🔮 브리즈번 조류량 예측 페이지')

    if 'predicted_chlorophyll' not in df.columns:
        st.error("예측 컬럼(predicted_chlorophyll)이 존재하지 않습니다. 모델 예측 결과를 추가해주세요.")
    else:
        fig = px.line(df, x='date', y='predicted_chlorophyll', title='예측 조류량 추세')
        st.plotly_chart(fig, use_container_width=True)

        selected_date = st.sidebar.date_input('날짜 선택')
        selected_row = df[df['date'] == str(selected_date)]

        if not selected_row.empty:
            value = selected_row['predicted_chlorophyll'].values[0]
            st.subheader(f"📅 {selected_date} 예측 조류량: {value:.2f}")

            threshold = 80
            if value > threshold:
                st.error('⚠️ 위험: 예측 조류량이 위험 수치를 초과했습니다.')
            else:
                st.success('🟢 안전: 예측 조류량이 안전 범위 안에 있습니다.')
        else:
            st.write('선택한 날짜의 예측 데이터가 없습니다.')