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

# 사이드바 - 파일 업로드
with st.sidebar:
    st.header("📁 데이터 업로드")
    uploaded_file = st.file_uploader(
        "CSV 또는 Excel 파일을 업로드하세요",
        type=['csv', 'xlsx', 'xls']
    )
# 메인 영역
if uploaded_file is not None:
    # 파일 타입에 따라 읽기
    # try:
    #     if uploaded_file.name.endswith('.csv'):
    #         # 인코딩 자동 감지 시도
    #         try:
    #             df = pd.read_csv(uploaded_file, encoding='utf-8')
    #         except UnicodeDecodeError:
    #             uploaded_file.seek(0)  # 파일 포인터 초기화
    #             df = pd.read_csv(uploaded_file, encoding='cp949')
    #     else:
    #         df = pd.read_excel(uploaded_file)
        
    #     # 날짜 컬럼 변환
    #     # if 'date' in df.columns:
    #     #     df['date'] = pd.to_datetime(df['date'], errors='coerce')
    #     #     df['month'] = df['date'].dt.to_period('M').astype(str)
    #     #     df['year_month'] = df['date'].dt.strftime('%Y-%m')
    #     if 'date' in df.columns:
    #         df['date'] = parse_mixed_date(df['date'])

    #         # 파생 변수
    #         df['year_month'] = df['date'].dt.to_period('M')
    #         df['weekday'] = df['date'].dt.day_name()

        
    #     if 'amount' in df.columns:
    #         # df['amount'] = df['amount'].astype(str).str.replace(',', '')
    #         df['amount'] = df['amount'].astype(str).str.strip()
    #         #  통화기호/한글/콤마/공백 제거 
    #         df['amount'] = df['amount'].str.replace(r'[^0-9\.\-]', '', regex=True)
    #         df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    #     df["category"] = df["category"].fillna("").replace("", "(미분류)")
    try:
        df = load_and_preprocess(uploaded_file)
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

# 핵심 지표 카드
st.markdown("### 📊 핵심 지표")
col1, col2, col3, col4 = st.columns(4)

total_expense = df_filtered['amount'].sum()
avg_expense = df_filtered['amount'].mean()
max_expense = df_filtered['amount'].max()
transaction_count = len(df_filtered)

col1.metric("💵 총 지출", f"{total_expense:,.0f}원")
col2.metric("📊 평균 지출", f"{avg_expense:,.0f}원")
col3.metric("📈 최대 지출", f"{max_expense:,.0f}원")
col4.metric("🧾 거래 건수", f"{transaction_count}건")

st.markdown("---")

# tab으로 분할
tab_viz, tab_ai, tab_report = st.tabs(["📊 시각화", "🤖 AI 인사이트","월간 리포트"])

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
                color_discrete_sequence=px.colors.qualitative.Set3
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
                markers=True
            )
            fig_line.update_layout(
                xaxis_title="월",
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
            color_continuous_scale='Oranges'
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

    # ✅ 텍스트는 trace에 직접 넣기
    fig.update_traces(
        text=text_pct,
        texttemplate="%{text}",
        textfont_size=12
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
    # ai 분석 기능
    def generate_expense_summary(df):
        """지출 데이터 요약 통계 생성"""
        summary = {
            'total': df['amount'].sum(),
            'average': df['amount'].mean(),
            'max': df['amount'].max(),
            'min': df['amount'].min(),
            'count': len(df),
        }
        
        # 카테고리별 통계
        if 'category' in df.columns:
            category_stats = df.groupby('category')['amount'].agg(['sum', 'count']).reset_index()
            category_stats['percentage'] = (category_stats['sum'] / summary['total'] * 100).round(1)
            summary['category_breakdown'] = category_stats.to_dict('records')
        
        # 월별 통계
        if 'year_month' in df.columns:
            monthly_stats = df.groupby('year_month')['amount'].sum().to_dict()
            summary['monthly'] = monthly_stats
        
        return summary
    
        # 프롬프트에 추가할 기간 값
    start_date = df_filtered['date'].min().strftime("%Y-%m-%d")
    end_date = df_filtered['date'].max().strftime("%Y-%m-%d")
    analysis_days = (df_filtered['date'].max() - df_filtered['date'].min()).days + 1

    def get_ai_insights(summary_data):
        """AI 인사이트 생성"""
        
        # 카테고리 breakdown을 문자열로 변환
        category_text = ""
        if 'category_breakdown' in summary_data:
            for item in summary_data['category_breakdown']:
                category_text += f"- {item['category']}: {item['sum']:,.0f}원 ({item['percentage']}%)\n"
        

        prompt = f"""
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
    """
        
        try:
            response = client.responses.create(
                model="gpt-4.1-mini",   # 또는 gpt-5-mini
                input=prompt,
                temperature=0.7,
                max_output_tokens=1000,
            )
            return response.output_text

        except Exception as e:
            return f"AI 분석 중 오류가 발생했습니다: {e}"

    # 초기화(한 번만)
    if "current_insights" not in st.session_state:
        st.session_state["current_insights"] = None
    if "prev_insights" not in st.session_state:
        st.session_state["prev_insights"] = None
    
    # Streamlit UI에서 사용
    st.markdown("---")
    st.markdown("### 🤖 AI 분석 인사이트")
    if client is None:
        st.warning("⚠️ OpenAI API 키가 설정되지 않았습니다.")
        st.info("Streamlit Cloud에서는 Secrets에 OPENAI_API_KEY를 추가해주세요.")
        st.stop()
        
    if st.button("🔍 AI 분석 시작", type="primary"):
        with st.spinner("AI가 지출 패턴을 분석하고 있습니다..."):
            summary = generate_expense_summary(df_filtered)
            insights = get_ai_insights(summary)
            st.session_state["last_summary"] = summary
            # st.markdown(insights)
            
            # # 분석 결과 저장
            # st.session_state['last_insights'] = insights
            # ✅ 새 분석 전에 기존 current를 prev로 넘기기
            if st.session_state["current_insights"]:
                st.session_state["prev_insights"] = st.session_state["current_insights"]

            # ✅ 새 결과는 current에 저장
            st.session_state["current_insights"] = insights

    # 이전 분석 결과 표시
    # if 'last_insights' in st.session_state:
    #     with st.expander("📝 이전 분석 결과 보기"):
    #         st.markdown(st.session_state['last_insights'])
    # ✅ 현재(최신) 결과 표시
    if st.session_state["current_insights"]:
        st.markdown(st.session_state["current_insights"])
    
    insights_text = st.session_state.get("current_insights")   # 또는 last_insights
    summary = st.session_state.get("last_summary")

    if insights_text:
        st.markdown("#### 📥 리포트 다운로드")

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fname = f"expense_ai_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        # 요약 섹션 문자열 만들기
        summary_md = ""
        if summary:
            summary_md = f"""
    - 총 지출: {summary['total']:,.0f}원
    - 평균 지출: {summary['average']:,.0f}원
    - 최대 지출: {summary['max']:,.0f}원
    - 최소 지출: {summary['min']:,.0f}원
    - 거래 건수: {summary['count']}건
    """
            # 카테고리 breakdown 있으면 표로 추가(선택)
            if "category_breakdown" in summary:
                summary_md += "\n\n### 카테고리별 지출\n\n| 카테고리 | 합계 | 비율 |\n|---|---:|---:|\n"
                for item in summary["category_breakdown"]:
                    summary_md += f"| {item['category']} | {item['sum']:,.0f}원 | {item['percentage']}% |\n"

        report_md = f"""# 🤖 AI 지출 분석 리포트

    생성일: {now}

    ---

    ## 1) 요약 통계
    {summary_md if summary_md else "(요약 통계가 없습니다)"}

    ---

    ## 2) AI 인사이트
    {insights_text}
    """

        st.download_button(
            label="📄 리포트 다운로드 (Markdown)",
            data=report_md,
            file_name=fname,
            mime="text/markdown"
        )
    else:
        st.info("AI 분석을 실행하면 리포트 다운로드가 활성화됩니다.")
    # ✅ 이전 결과는 별도로 표시 (새 분석해도 여기 값은 '직전'으로만 갱신)
    if st.session_state["prev_insights"]:
        with st.expander("📝 이전 분석 결과 보기"):
            st.markdown(st.session_state["prev_insights"])

with tab_report:
    def generate_monthly_report(df, insights=None):
        """월간 리포트 마크다운 생성"""
        
        report = f"""
    #  월간 지출 리포트

    생성일: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

    ---

    ## 📈 지출 요약

    | 항목 | 금액 |
    |------|------|
    | 총 지출 | {df['amount'].sum():,.0f}원 |
    | 평균 지출 | {df['amount'].mean():,.0f}원 |
    | 최대 지출 | {df['amount'].max():,.0f}원 |
    | 거래 건수 | {len(df)}건 |

    ---
    
    #  카테고리별 지출
        """
    
    
        
        if 'category' in df.columns:
            category_sum = df.groupby('category')['amount'].sum().sort_values(ascending=False)
            total = category_sum.sum()
            
            report += "\n| 카테고리 | 금액 | 비율 |\n"
            report += "|----------|------|------|\n"
            for cat, amount in category_sum.items():
                percentage = (amount / total * 100)
                report += f"| {cat} | {amount:,.0f}원 | {percentage:.1f}% |\n" 
        
        report += "\n---\n\n##  상위 5개 지출\n\n"
        
        top5 = df.nlargest(5, 'amount')[['date', 'category', 'description', 'amount']]
        report += "| 날짜 | 카테고리 | 내용 | 금액 |\n"
        report += "|------|----------|------|------|\n"
        for _, row in top5.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else '-'
            report += f"| {date_str} | {row['category']} | {row['description']} | {row['amount']:,.0f}원 |\n"
        
        if insights:
            report += f"\n---\n\n## 🤖 AI 인사이트\n\n{insights}\n"
        
        return textwrap.dedent(report).strip()

    # Streamlit UI에서 사용
    st.markdown("---")
    st.markdown("### 📋 월간 리포트")

    if st.button("📄 리포트 생성"):
        insights = st.session_state.get('current_insights', None)
        report = generate_monthly_report(df_filtered, insights)
        
        st.markdown(report)
        
        # 다운로드 버튼
        st.download_button(
            label="📥 리포트 다운로드 (Markdown)",
            data=report,
            file_name=f"expense_report_{pd.Timestamp.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )