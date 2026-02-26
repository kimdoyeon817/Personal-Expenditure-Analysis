import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
import datetime
import textwrap

#client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
def get_client():
    if "OPENAI_API_KEY" not in st.secrets:
        return None
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

client = get_client()

st.set_page_config(
    page_title=" 개인 지출 분석 대시보드",
    page_icon="💰",
    initial_sidebar_state= "collapsed", # 사이드 바 닫힘 상태
    layout="wide" 
)

st.title("💰 개인 지출 분석 대시보드")

# date 일반 포맷 자동 파싱
def parse_mixed_date(series: pd.Series) -> pd.Series:
    # 1차: 일반적인 날짜 포맷 자동 파싱
    parsed = pd.to_datetime(series, errors='coerce')

    # 2차: YYYYMMDD 형태만 골라서 재파싱
    mask = parsed.isna() & series.astype(str).str.match(r'^\d{8}$')
    parsed.loc[mask] = pd.to_datetime(
        series[mask],
        format='%Y%m%d',
        errors='coerce'
    )

    return parsed

@st.cache_data(show_spinner=False)
def load_and_preprocess(uploaded_file) -> pd.DataFrame:
    # 파일 타입에 따라 읽기
    if uploaded_file.name.endswith('.csv'):
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='cp949')
    else:
        df = pd.read_excel(uploaded_file)

    # 날짜 파싱 + 파생 변수
    if 'date' in df.columns:
        df['date'] = parse_mixed_date(df['date'])
        df['year_month'] = df['date'].dt.to_period('M')   # 문자열 대신 Period
        df['weekday'] = df['date'].dt.day_name()

    # amount 파싱(강화)
    if 'amount' in df.columns:
        df['amount'] = df['amount'].astype(str).str.strip()
        df['amount'] = df['amount'].str.replace(r'^\((.*)\)$', r'-\1', regex=True)
        # 통화기호/한글/콤마/공백 제거: 숫자/소수점/부호만 남김
        df['amount'] = df['amount'].str.replace(r'[^0-9\.\-]', '', regex=True)
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    # category 정리
    if 'category' in df.columns:
        df["category"] = df["category"].fillna("").replace("", "(미분류)")

    return df

# =========================
#  공용 AI 함수 (전역)
# =========================
def generate_expense_summary(df: pd.DataFrame) -> dict:
    summary = {
        "total": df["amount"].sum(),
        "average": df["amount"].mean(),
        "max": df["amount"].max(),
        "min": df["amount"].min(),
        "count": len(df),
    }

    if "category" in df.columns and summary["total"] and summary["total"] != 0:
        category_stats = (
            df.groupby("category")["amount"]
            .agg(["sum", "count"])
            .reset_index()
            .sort_values("sum", ascending=False)
        )
        category_stats["percentage"] = (category_stats["sum"] / summary["total"] * 100).round(1)
        summary["category_breakdown"] = category_stats.to_dict("records")

    if "year_month" in df.columns:
        summary["monthly"] = df.groupby("year_month")["amount"].sum().to_dict()

    return summary


def build_prompt(summary_data: dict, start_date: str, end_date: str, analysis_days: int) -> str:
    category_text = ""
    if "category_breakdown" in summary_data:
        for item in summary_data["category_breakdown"]:
            category_text += f"- {item['category']}: {item['sum']:,.0f}원 ({item['percentage']}%)\n"

    return f"""
    당신은 10년 경력의 개인 재무 컨설턴트입니다.
    아래 기간 동안의 소비 데이터를 분석하여 구조화된 재무 리포트를 작성해주세요.
    
    [분석 기간]
    - 시작일: {start_date}
    - 종료일: {end_date}
    - 총 분석 기간: {analysis_days}일

    [지출 요약]
    - 총 지출: {summary_data['total']:,.0f}원
    - 평균 지출: {summary_data['average']:,.0f}원
    - 최대 단일 지출: {summary_data['max']:,.0f}원
    - 거래 건수: {summary_data['count']}건

    [카테고리별 지출]
    {category_text}

    [분석 요청]
    1. 소비 패턴 분석
    - 분석 기간을 고려하여 소비 규모 평가
    - 일 평균 지출 수준이 적정한지 판단
    - 과소비 카테고리 명확히 제시

    2. 절약 가능 영역 제안
    - 절약 가능한 카테고리
    - 월 기준 예상 절감 금액 제시
    - 구체적인 행동 방법 포함

    3. 다음 달 권장 예산
    - 카테고리별 권장 월 예산 제시
    - 전체 목표 월 예산 제시
    - 관리 전략 1~2줄 요약

    조건:
    - 반드시 수치를 근거로 설명
    - 모호한 표현 금지
    - 500~800자 내 작성
    - 보고서 형태 유지
    """.strip()


def get_ai_insights_for_df(df: pd.DataFrame, client: OpenAI) -> tuple[str, dict]:
    """df 기준으로 AI 인사이트 생성. (insights_text, summary) 반환"""
    summary = generate_expense_summary(df)

    if "date" in df.columns and df["date"].notna().any():
        start_date = df["date"].min().strftime("%Y-%m-%d")
        end_date = df["date"].max().strftime("%Y-%m-%d")
        analysis_days = (df["date"].max() - df["date"].min()).days + 1
    else:
        start_date, end_date, analysis_days = "-", "-", 0

    prompt = build_prompt(summary, start_date, end_date, analysis_days)

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0.7,
            max_output_tokens=1000,
        )
        return response.output_text, summary
    except Exception as e:
        return f"⚠ AI 분석 중 오류가 발생했습니다: {str(e)}", summary

# 사이드바 - 파일 업로드
with st.sidebar:
    st.header("📁 데이터 업로드")
    uploaded_file = st.file_uploader(
        "CSV 또는 Excel 파일을 업로드하세요",
        type=['csv', 'xlsx', 'xls']
    )
# 메인 영역
if uploaded_file is not None:
    
    try:
        df = load_and_preprocess(uploaded_file)
        required_columns = {"amount", "date", "category"}

        missing = required_columns - set(df.columns)

        if missing:
            st.error(f"""
잘못된 파일 형식입니다.

필수 컬럼:
- amount
- date
- category

누락된 컬럼: {', '.join(missing)}
        """)
            st.stop()

        st.session_state["df"] = df
        st.session_state["uploaded_name"] = uploaded_file.name

        st.success(f"✅ 데이터 로드 완료! ({len(df)}건)")
        
        # 데이터 미리보기
        with st.expander("📋 데이터 미리보기"):
            st.dataframe(df.head(10))
        
    except Exception as e:
        st.error(f"파일 읽기 오류: {e}")
        
else:
    st.info("👈 왼쪽 사이드바에서 파일을 업로드해주세요.")
    
    # 샘플 데이터 다운로드 버튼
    st.markdown("---")
    st.markdown("### 📥 샘플 데이터가 필요하신가요?")
    
    # 샘플 데이터 생성
    sample_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=30, freq='D'),
        'amount': [15000, 3500, 45000, 12000, 8500, 25000, 6000, 
                   32000, 4500, 18000, 55000, 7500, 21000, 9000,
                   28000, 5500, 16000, 42000, 11000, 8000, 35000,
                   4000, 22000, 13500, 48000, 6500, 19000, 38000,
                   7000, 26000],
        'category': ['식비', '교통비', '쇼핑', '식비', '카페', '문화',
                     '교통비', '식비', '카페', '쇼핑', '의료', '교통비',
                     '식비', '카페', '쇼핑', '교통비', '식비', '문화',
                     '교통비', '카페', '식비', '교통비', '쇼핑', '식비',
                     '문화', '카페', '식비', '쇼핑', '교통비', '식비'],
        'description': ['점심 식사', '지하철', '옷 구매', '저녁 식사', '커피',
                        '영화', '버스', '회식', '아메리카노', '온라인쇼핑',
                        '병원', '택시', '배달음식', '카페라떼', '생필품',
                        '지하철', '편의점', '콘서트', '버스', '디저트',
                        '장보기', '지하철', '신발', '외식', '전시회',
                        '커피', '점심', '악세서리', '택시', '저녁']
    })
    
    csv = sample_data.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 샘플 CSV 다운로드",
        data=csv,
        file_name="sample_expense_data.csv",
        mime="text/csv"
    )

# 이전 코드에 이어서...

# if uploaded_file is not None and 'df' in dir():
df = st.session_state.get("df")
if df is None:
    st.stop()    
# 사이드바 - 필터
with st.sidebar:
    st.header("🔍 필터")
    
    # 기간 필터
    if 'date' in df.columns:
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        date_range = st.date_input(
            "기간 선택",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df[
                (df['date'].dt.date >= start_date) & 
                (df['date'].dt.date <= end_date)
            ]
        else:
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()
    
    # 카테고리 필터
    if 'category' in df.columns:
        categories = df['category'].unique().tolist()
        selected_categories = st.multiselect(
            "카테고리 선택",
            options=categories,
            default=categories
        )
        df_filtered = df_filtered[df_filtered['category'].isin(selected_categories)]
        
    if df_filtered.empty:
        st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
        st.stop()

# 핵심 지표 카드
st.markdown("### 📊 핵심 지표")
col1, col2, col3, col4 = st.columns(4)

total_expense = df_filtered['amount'].sum()
avg_expense = df_filtered['amount'].mean()
max_expense = df_filtered['amount'].max()
transaction_count = len(df_filtered)

col1.metric(" 총 지출", f"{total_expense:,.0f}원")
col2.metric(" 평균 지출", f"{avg_expense:,.0f}원")
col3.metric(" 최대 지출", f"{max_expense:,.0f}원")
col4.metric(" 거래 건수", f"{transaction_count}건")

st.markdown("---")

# tab으로 분할
tab_viz, tab_ai, tab_report = st.tabs(["📊 시각화", "🤖 AI 인사이트","📄 월간 리포트"])
# =========================
#  세션 상태 초기화 (전체 AI / 월간 AI 분리)
# =========================
st.session_state.setdefault("ai_global_current", None)
st.session_state.setdefault("ai_global_prev", None)
st.session_state.setdefault("ai_global_summary", None)

# 월별 저장 (선택 월이 바뀌어도 결과 유지)
st.session_state.setdefault("ai_monthly", {})          # { "2024-02": "insights text" }
st.session_state.setdefault("ai_monthly_summary", {})  # { "2024-02": summary dict }


with tab_viz:
    # 차트 영역
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 🥧 카테고리별 지출")
        if 'category' in df_filtered.columns:
            category_sum = df_filtered.groupby('category')['amount'].sum().reset_index()
            fig_pie = px.pie(
                category_sum, 
                values='amount', 
                names='category',
                hole=0.4,  # 도넛 차트
                color_discrete_sequence=px.colors.qualitative.Set3,
                labels={
                    "category": "카테고리",
                    "amount": "지출 금액 (원)"
                }
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_right:
        st.markdown("### 📈 월별 지출 추이")
        if 'year_month' in df_filtered.columns:
            monthly_sum = df_filtered.groupby('year_month', as_index=False)['amount'].sum().sort_values('year_month')
            monthly_sum['year_month_str'] = monthly_sum['year_month'].astype(str)
            fig_line = px.line(
                monthly_sum, 
                x='year_month_str', 
                y='amount',
                markers=True,
                labels={
                    "year_month_str": "날짜",
                    "amount": "지출 금액 (원)"
                }
            )
            fig_line.update_layout(
                xaxis_title="날짜",
                yaxis_title="지출 금액 (원)"
            )
            st.plotly_chart(fig_line, use_container_width=True)
    
    # 카테고리별 바 차트
    st.markdown("### 📊 카테고리별 지출 금액")
    if 'category' in df_filtered.columns:
        category_sum = df_filtered.groupby('category')['amount'].sum().reset_index()
        category_sum = category_sum.sort_values('amount', ascending=True)
        
        fig_bar = px.bar(
            category_sum,
            x='amount',
            y='category',
            orientation='h',
            color='amount',
            color_continuous_scale='Oranges',
            labels={
                    "category": "카테고리",
                    "amount": "지출 금액 (원)"
                }
        )
        fig_bar.update_layout(
            xaxis_title="지출 금액 (원)",
            yaxis_title="카테고리",
            showlegend=False,
            xaxis_tickformat=","
        )
        st.plotly_chart(fig_bar, use_container_width=True)


    st.markdown("### 🔥 요일별 지출 패턴 (비율 %)")

    df_heat = df_filtered.copy()

    df_heat["weekday_kr"] = df_heat["date"].dt.dayofweek.map({
        0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"
    })
    df_heat["category"] = df_heat["category"].fillna("").replace("", "(미분류)")

    # amount 숫자형 보정
    # df_heat["amount"] = df_heat["amount"].astype(str).str.replace(",", "", regex=False).str.strip()
    # df_heat["amount"] = pd.to_numeric(df_heat["amount"], errors="coerce").fillna(0)
    df_heat["amount"] = pd.to_numeric(df_heat["amount"], errors="coerce").fillna(0)
    weekday_order = ["월", "화", "수", "목", "금", "토", "일"]

    # 1) 금액 pivot
    pivot = (
        df_heat.pivot_table(
            index="category",
            columns="weekday_kr",
            values="amount",
            aggfunc="sum",
            fill_value=0
        ).reindex(columns=weekday_order)
    )

    # 2) 비율 pivot (행=카테고리 기준)
    row_sum = pivot.sum(axis=1).replace(0, 1)   # 0으로 나누기 방지
    pivot_pct = pivot.div(row_sum, axis=0) * 100

    # 3) 셀 텍스트 (%)
    text_pct = pivot_pct.applymap(lambda v: f"{v:.1f}%" if v > 0 else "").to_numpy()

    # 4) 히트맵
    fig = px.imshow(
        pivot_pct,                          # 값(%)은 DataFrame 그대로 OK
        aspect="auto",
        color_continuous_scale="Blues",
        zmin=0, zmax=100
    )

    #  텍스트는 trace에 직접 넣기
    fig.update_traces(
        text=text_pct,
        texttemplate="%{text}",
        textfont_size=12,
        hovertemplate="카테고리 = %{y}<br>"
                  "요일 =  %{x}<br>"
                  "지출 비율 = %{z:.1f}%<extra></extra>"
    )

    fig.update_layout(
        xaxis_title="요일",
        yaxis_title="카테고리",
        coloraxis_colorbar=dict(
            title="지출 비율 (%)",
            ticksuffix="%"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

with tab_ai:
    
    st.markdown("---")
    st.markdown("### 🤖 AI 분석 인사이트 (전체/필터 기간)")

    if client is None:
        st.warning("⚠️ OpenAI API 키가 설정되지 않았습니다.")
        st.info("Streamlit Cloud에서는 Secrets에 OPENAI_API_KEY를 추가해주세요.")
        st.stop()

    if st.button("🔍 전체 기간 AI 분석 시작", type="primary"):
        now = datetime.datetime.now()

        if "last_ai_call" in st.session_state:
            last_call = st.session_state["last_ai_call"]
            if (now - last_call).total_seconds() < 10:
                st.warning("⚠️ 잠시 후 다시 시도해주세요. (10초 제한)")
                st.stop()

        st.session_state["last_ai_call"] = now

        with st.spinner("AI가 전체 기간 지출 패턴을 분석하고 있습니다..."):
            insights, summary = get_ai_insights_for_df(df_filtered, client)

            if st.session_state["ai_global_current"]:
                st.session_state["ai_global_prev"] = st.session_state["ai_global_current"]

            st.session_state["ai_global_current"] = insights
            st.session_state["ai_global_summary"] = summary

    if st.session_state.get("ai_global_current"):
        st.markdown(st.session_state["ai_global_current"])
    else:
        st.info("전체 기간 AI 분석을 실행하면 결과가 표시됩니다.")

    if st.session_state.get("ai_global_prev"):
        with st.expander("📝 이전(직전) 전체 분석 결과 보기"):
            st.markdown(st.session_state["ai_global_prev"])

    current_insights = st.session_state.get("ai_global_current")
    current_summary = st.session_state.get("ai_global_summary")

    if current_insights:
        st.markdown("#### 📥 리포트 다운로드")

        # now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # fname = f"expense_ai_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        now_dt = datetime.datetime.now()
        now = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        fname = f"expense_ai_report_{now_dt.strftime('%Y%m%d_%H%M%S')}.md"

        # 요약 섹션 문자열 만들기
        summary_md = ""
        if current_summary:
            summary_md = f"""
- 총 지출: {current_summary['total']:,.0f}원
- 평균 지출: {current_summary['average']:,.0f}원
- 최대 지출: {current_summary['max']:,.0f}원
- 최소 지출: {current_summary['min']:,.0f}원
- 거래 건수: {current_summary['count']}건
"""
            # 카테고리 breakdown 있으면 표로 추가(선택)
            if "category_breakdown" in current_summary:
                summary_md += "\n\n### 카테고리별 지출\n\n| 카테고리 | 합계 | 비율 |\n|---|---:|---:|\n"
                for item in current_summary["category_breakdown"]:
                    summary_md += f"| {item['category']} | {item['sum']:,.0f}원 | {item['percentage']}% |\n"

        report_md = f"""# 🤖 AI 지출 분석 리포트

    생성일: {now}

    ---

    ## 1) 요약 통계
    {summary_md if summary_md else "(요약 통계가 없습니다)"}

    ---

    ## 2) AI 인사이트
    {current_insights}
    """

        st.download_button(
            label="📄 리포트 다운로드 (Markdown)",
            data=report_md,
            file_name=fname,
            mime="text/markdown"
        )

with tab_report:

    # -------------------------
    # 1) 월 선택
    # -------------------------
    st.markdown("---")
    st.markdown("### 📋 월간 리포트")

    if "year_month" in df_filtered.columns:
        month_options = sorted(df_filtered["year_month"].dropna().unique())
        selected_month = st.selectbox(
            "📅 리포트 생성 월 선택",
            options=month_options,
            format_func=lambda x: str(x),
        )
        month_key = str(selected_month)
        df_month = df_filtered[df_filtered["year_month"] == selected_month].copy()
    else:
        month_key = "selected"
        df_month = df_filtered.copy()

    if df_month.empty:
        st.warning("선택한 월에 데이터가 없습니다.")
        st.stop()

    # -------------------------
    # 2) 리포트 생성 함수 (문자열만 생성)
    # -------------------------
    def generate_monthly_report(df, insights=None):

        month_label = (
            str(df["year_month"].iloc[0])
            if "year_month" in df.columns and not df.empty
            else "선택 월"
        )

        report = f"""
# {month_label} 월 지출 리포트
생성일: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

---

## 지출 요약

| 항목 | 금액 |
|------|------|
| 총 지출 | {df['amount'].sum():,.0f}원 |
| 평균 지출 | {df['amount'].mean():,.0f}원 |
| 최대 지출 | {df['amount'].max():,.0f}원 |
| 거래 건수 | {len(df)}건 |

---

## 카테고리별 지출
"""

        if "category" in df.columns:
            category_sum = (
                df.groupby("category")["amount"]
                .sum()
                .sort_values(ascending=False)
            )
            total = category_sum.sum()

            report += "\n| 카테고리 | 금액 | 비율 |\n"
            report += "|----------|------|------|\n"

            for cat, amount in category_sum.items():
                percentage = (amount / total * 100) if total else 0
                report += f"| {cat} | {amount:,.0f}원 | {percentage:.1f}% |\n"

        report += "\n---\n\n## 상위 5개 지출\n\n"

        # top5 = df.nlargest(5, "amount")[["date", "category", "description", "amount"]]
        cols = ["date", "category", "amount"]
        if "description" in df.columns:
            cols.insert(2, "description")

        top5 = df.nlargest(5, "amount")[cols]
        report += "| 날짜 | 카테고리 | 내용 | 금액 |\n"
        report += "|------|----------|------|------|\n"

        for _, row in top5.iterrows():
            date_str = row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else "-"
            desc = row["description"] if "description" in row and pd.notna(row["description"]) else "-"
            report += f"| {date_str} | {row['category']} | {desc} | {row['amount']:,.0f}원 |\n"

        report += "\n---\n\n## 🤖 월간 AI 인사이트\n\n"

        if insights:
            report += f"{insights}\n"
        else:
            report += "⚠ OpenAI API 키가 없거나 AI 인사이트 생성에 실패했습니다.\n"

        return textwrap.dedent(report).strip()

    # -------------------------
    # 3) 버튼 하나만: 누르면 AI 생성 + 리포트 출력
    # -------------------------
    if st.button("📄 월간 리포트 생성", type="primary"):

        insights_text = None
        now = datetime.datetime.now()

        if "last_ai_call" in st.session_state:
            last_call = st.session_state["last_ai_call"]
            if (now - last_call).total_seconds() < 10:
                st.warning("⚠️ 잠시 후 다시 시도해주세요. (10초 제한)")
                st.stop()

        st.session_state["last_ai_call"] = now

        if client is None:
            st.warning("⚠️ OpenAI API 키가 설정되지 않아 AI 인사이트 없이 리포트를 생성합니다.")
        else:
            cached = st.session_state.get("ai_monthly", {}).get(month_key)

            if cached:
                insights_text = cached
            else:
                with st.spinner(f"{month_key} 월 AI 인사이트 생성 중..."):
                    try:
                        insights_text, summary = get_ai_insights_for_df(df_month, client)
                        st.session_state.setdefault("ai_monthly", {})
                        st.session_state.setdefault("ai_monthly_summary", {})
                        st.session_state["ai_monthly"][month_key] = insights_text
                        st.session_state["ai_monthly_summary"][month_key] = summary
                    except Exception as e:
                        st.error(f"AI 인사이트 생성 중 오류: {e}")
                        insights_text = None

        report = generate_monthly_report(df_month, insights_text)

        st.markdown(report)

        st.download_button(
            label="📥 리포트 다운로드 (Markdown)",
            data=report,
            file_name=f"expense_report_{month_key}.md",
            mime="text/markdown",
        )