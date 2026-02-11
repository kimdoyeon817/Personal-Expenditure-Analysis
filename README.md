# AI 기반 개인 지출 패턴 분석 대시보드

##  프로젝트 개요

사용자의 지출 데이터를 업로드하면 자동으로 전처리 및 분석을 수행하고,  
소비 패턴을 시각화한 뒤 OpenAI API를 활용해  
AI 기반 소비 인사이트 및 맞춤형 예산을 추천하는 웹 대시보드입니다.

> "내 지출 데이터를 넣으면, AI가 소비 습관을 분석해주는 개인 재무 코치"

---

## 프로젝트 목적

- 소비 데이터를 단순 숫자가 아닌 **패턴 기반 정보**로 전환
- 과소비 영역을 시각적으로 식별
- AI 기반 예산 추천을 통한 데이터 중심 소비 관리 지원
- 전처리 → 시각화 → AI 분석까지 하나의 분석 파이프라인 구현

---

##  실행 방법

### 가상환경 생성 및 활성화

```bash
python -m venv venv
.\venv\Scripts\activate

### 패키지 설치
pip install -r requirements.txt
### Streamlit 실행
streamlit run app.py

## OpenAI API Key 설정

프로젝트 루트에 .streamlit 폴더 생성 후
secrets.toml 파일을 아래 형식으로 작성:

.streamlit/
    secrets.toml

OPENAI_API_KEY="your-api-key-here"


## 시스템 흐름

사용자 파일 업로드
        ↓
데이터 전처리
        ↓
카테고리 분류
        ↓
집계 지표 계산
        ↓
차트 시각화
        ↓
AI 분석 요청
        ↓
인사이트 및 예산 추천 출력
