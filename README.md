# 프로젝트 전체 개요

## Project : 내방 어디?
![image](https://user-images.githubusercontent.com/120995529/216347597-1ff2140e-78d4-424d-8ffc-4ff9fdc269f9.png)

## 목적
- 서울시 전/월세 실거래 데이터를 기반한 검색, 머신러닝을 이용한 전세 시세 예측
***

## 데이터
- [서울특별시 부동산 전월세가 정보](https://data.seoul.go.kr/dataList/OA-21276/S/1/datasetView.do)
***

## 링크
- 폴더 링크 : [Project_1](https://github.com/Depra3/Human_Project1)
- 배포 링크 : [내 방, 어디?](https://depra3-human-project1-app-kbirqs.streamlit.app/)
- 시연 영상 링크 : [시연 영상](https://www.youtube.com/watch?v=RbN39pUA1Ww&ab_channel=evanjjh)
- PDF : 
***

## Project 개요
- 인원 : 4명
- 기간 : `2023.01.27 ~ 2023.02.10`
- 사용 언어 : <img src="https://img.shields.io/badge/Python:3.9.13-3776AB?style=flat&logo=Python&logoColor=white"/>
- 사용 Tool : <img src="https://img.shields.io/badge/Python:1.74.1-3776AB?style=flat&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/Google Colab-F9AB00?style=flat&logo=Google Colab&logoColor=white"/>
- 사용 DB : <img src="https://img.shields.io/badge/SQLite:2.6.0-003B57?style=flat&logo=SQLite&logoColor=white"/>
- 라이브러리 `Version`
    + `pandas (1.5.3)`, `numpy (1.24.1)`, `plotly (5.13.0)`, `matplotlib (3.6.3)`, `streamlit (1.17.0)`, `streamlit-option-menu (0.3.2)`, `geopandas (0.12.2)`, `joblib (1.2.0)`, `scikit-learn (1.2.1)`, `tensorflow (2.9.0)`, `seaborn (0.12.2)`, `geopandas (0.12.2)`, `pydeck (0.8.0)`, `prophet (1.1.2)`, `openai (0.26.5)`, `streamlit_chat (0.0.2.1)`, `requests (2.28.2)`,`sqlite(2.6.0)`
***

## 기간 상세

<details>
<summary><h4>2023/01/27 ~ 2023/02/10</h4></summary>
<div markdown="1">

<details>
<summary><h4>2023/01/27</h4></summary>
<div markdown="1">

- 주제 선정
    + 부동산 전/월세 관련 내용
</div>
</details>

<details>
<summary><h4>2023/01/30</h4></summary>
<div markdown="1">

- 개발팀
    + 오늘 한 내용
        - streamlit 배포
        - 전체적인 스토리 구상
        - 기본적인 UI 구현
    + 오늘 못 한 내용
        - 추천 기능 시도 : 거래 매물이 많은 지역 (or 면적당 가격이 싼 곳)
        - 전, 월세 검색 페이지 구현
        - 건의사항 페이지 구현
    + 내일 할 내용
        - 전/월세 검색 및 비교 기능 추가
        - 결과물 동기화

- 데이터팀
    + 오늘 한 내용
        - 공공데이터 검색
        - 결측치 제거, 불필요한 데이터 제거
        - 시나리오 작성
    + 오늘 못한 내용
        - 데이터 전처리
        - 시나리오 재작성
        - 데이터 시각화, 예측모델 구상
    + 내일 할 내용
        - 데이터 전처리
        - 데이터 시각화
        - 시나리오 재작성
</div>
</details>

<details>
<summary><h4>2023/01/31</h4></summary>
<div markdown="1">

- 데이터 처리 방법 선택
    + API를 이용하여 DB에 저장
- 웹개발 프레임워크 선택 (`Flask` or `Streamlit`)
    + 보다 쉬워보이는 `Streamlit`로 선택
- 전체 시나리오 구상
</div>
</details>

<details>
<summary><h4>2023/02/01</h4></summary>
<div markdown="1">

- 개발팀
    + 오늘 한 내용
        - Index 페이지 
            + 기본적인 layer 구상
            + 거래횟수가 많은 지역 순으로 데이터 정렬
        - 전/월세 페이지
            + Sidebar에서 조건에 맞는 검색 기능
            + 보증금, 월세, 면적의 최소값/최대값을 지정해주는 슬라이더
            + 버튼 누를 시 선택된 값에 해당하는 검색 기능
            + 면적 제곱미터를 평수로 변환하는 람다식
            + 필요한 칼럼을 조인하여 데이터 가공
            + 특정 칼럼에서 특정 문자 삭제
        - 건의사항 페이지
            + 게시판 UI 및 기능 구현
            + sqlite DB 연동
        - 코드 동기화
    + 오늘 못 한 내용
        - Index 페이지
            + 정렬한 데이터 추출        
        - 전월세 페이지
            + 전세와 월세를 동시에 보여주는 기능
            + 보증금 월세 범위가 예상보다 컸음
        - 건의사항 페이지
            + 게시글 수정&삭제 기능(추후에 html 활용하여 추가 예정)
    + 내일 할 내용
        - 데이터 추가 핸들링
        - home 페이지 디자인 마무리
        - 전세예측 페이지 구현
        - 게시글 수정 & 삭제 기능
        - 건의사항 내용 칸 늘리기

- 데이터팀
    + 오늘 한 내용
        - 데이터 전처리 정리
        - 최적 시각화 그래프 서칭
        - 프로젝트 시나리오 재작성
        - streamlit 사용하여 구, 동 선택 가능한multislectbox 구현
        - 구, 동 별 데이터 시각화 코드 작성
    + 오늘 못 한 내용
        - 데이터 전처리(일일 평균)
        - 그래프 최적화
    + 내일 할 내용
        - 구, 동 별 일일 평균 시각화 코드 작성
        - 막대 및 지도 시각화
        - 시나리오 보충
</div>
</details>

<details>
<summary><h4>2023/02/02</h4></summary>
<div markdown="1">

- 개발팀
    + 오늘 한 것
        - homepage UI 디자인변경
        - homepage dataframe구성 변경
        - 월/전세 전체 검색기능
        - 월세, 보증금, 면적검색할 때 최소, 최댓값 입력 기능
        - 건의사항 목록 간격 수정
        - 건의사항 처리상태 변경 기능
        - 건의사항 제목, 사용자명 검색 기능
    + 오늘 못한 것
        - 지역에 맞춘 keyword 알고리즘
        - 건의사항 게시글 조회, 수정, 삭제 기능
    + 내일 할 것
        - homepage keyword 알고리즘 구현
        - 건의사항 검색 UI 수정
        - 건의사항 내용 검색 기능(디버깅)
        - 건의사항 검색 시 목록 수정
        - 건의사항 목록 간격

- 데이터팀
    + 오늘 한 것
        - 데이터 전처리 정리
        - 최적 시각화 그래프 서칭
        - 지도 그래프 시각화 코드 작성(진행중)
        - 프로젝트 시나리오 재작성
        - streamlit 사용하여 구, 동 선택 가능한 multislectbox 구현
        - 구, 동 별 데이터 시각화 코드 작성
    + 오늘 못한 것
        - 데이터 전처리(일일 평균)
        - 지도 그래프 시각화 코드 작성(미완)
        - 그래프 최적화
    + 내일 할 것
        - 구, 동 별 일일 평균 시각화 코드 작성
        - 막대 및 지도 시각화
        - 시나리오 보충(도식화)
</div>
</details>

<details>
<summary><h4>2023/02/03</h4></summary>
<div markdown="1">

- 개발팀
    + 오늘 한 것
        - 검색 페이지 슬라이더 기능 조정
            + 슬라이더와 텍스트박스값 연동
        - 건의사항
            + 게시글 조회 기능
            + UI 변경
            + 관리자 메뉴 기능 추가
    + 오늘 못한 것
        - csv데이터값 결함 수정
            + 월/전세 구분 오류 확인
        - 건의사항 목록 간격 지정
    + 내일 할 것
        - csv데이터값 결함 수정
        - 건의사항 관리자 메뉴 숨기기

- 데이터팀
    + 오늘 한 것
        - json 파일 변경
        - csv파일과 json파일 병합
        - 지도 시각화 구현
        - 실거래가 머신러닝 코드 분석
        - 월세 실거래수 지역 순위 막대그래프 구현
        - 전세 실거래 수 지역 순위 막대그래프 구현
        - 전세 및 월세(보증금) 월 평균 라인그래프 구현
        - 실거래가 데이터 전처리 진행
    + 오늘 못한 것
        - 실거래가 머신러닝 코드 구현
        - 전세 및 월세 실거래가 데이터 전처리
        - 전세 및 월세 실거래가 계산 레이아웃 구현
    + 다음주 할 것
        - 실거래가 머신러닝 코드 구현
        - 전세 및 월세 실거래가 데이터 전처리
        - 전세 및 월세 실거래가 계산 레이아웃 구현
        - 지도 시각화 수정
</div>
</details>

<details>
<summary><h4>2023/02/06</h4></summary>
<div markdown="1">

- 개발팀
    + 오늘 한 것
        - BigQuery를 통한 data cloud 구현 시도
    + 오늘 못한 것
        - 실시간 데이터 수집(api로 받기)
    + 내일 할 것
        - 실시간 데이터 수집
        - 홈페이지 기능 재배치
- 데이터팀
    + 오늘 한 것
        - json 파일 geojson으로 변경
        - csv파일과 json파일 병합
        - 지도 시각화 구현(완료)
        - 실거래가 머신러닝 코드 분석(진행중)
        - 실거래가 데이터 전처리 진행(완료)
    + 오늘 못한 것
        - 실거래가 머신러닝 코드 구현(진행중)
        - 전세 및 월세 실거래가 계산 레이아웃 구현(진행중)
    + 내일 할 것
        - 실거래가 머신러닝 코드 구현
        - 전세 및 월세 실거래가 계산 레이아웃 구현
</div>
</details>

<details>
<summary><h4>2023/02/07</h4></summary>
<div markdown="1">

- 개발팀
    + 오늘 한 것
        - 홈페이지 재배치
        - 검색 slider에 scale 추가
        - API data를 DB에 연결
    + 오늘 못한 것
        - 실시간 데이터 수집
            + batch process 구현
    + 내일 할 것
        - batch process 구현

- 데이터팀
    + 오늘 한 것
        - 실거래가 머신러닝 코드 구현1(완료)
        - 실거래가 머신러닝 코드 분석(완료)
        - 전세 및 월세 실거래가 계산 레이아웃 구현(진행중)
    + 오늘 못한 것
        - 실거래가 머신러닝 코드 구현2(진행중)
        - 전세 및 월세 실거래가 계산 레이아웃 구현(진행중)
    + 내일 할 것
        - 실거래가 머신러닝 코드 구현2
        - 전세 및 월세 실거래가 계산 코드 작성
</div>
</details>

<details>
<summary><h4>2023/02/08</h4></summary>
<div markdown="1">

- 개발팀
    + 오늘 한 것
        - 배치파일을 이용한 실시간 데이터 수집
        - 실시간 데이터받는 중 서치.py에 버그 수정
            + (받아오는 데이터들이 StringType이여서 int형으로 변환)
        - 각 기능 설명 추가
        - 건의사항 UI 기능추가(진행중) 및 변경
    + 오늘 못한 것
        - 전월세 검색 페이지 버그 수정
    + 내일 할 것
        - 전월세 검색 페이지에 조회버튼 버그 수정
        - 배치파일을 이용해 update되는지 확인

- 데이터팀
    + 오늘 한 것
        - LSTM 예측 모델 보완
        - prophet 예측 모델 완성
        - 그래프 레이아웃 시각화 수정( 구 선택 추가 )
        - ChatBot 데이터 연동
        - 전월세 전환 계산 1 구현 ( 전세 --> 월세 )
    + 오늘 못한 것
        - 머신러닝 시각화 레이아웃 수정
        - 전월세 전환 계산 2,3 구현( 월세--> 전세, 전환율 계산)
    + 내일 할 것
        - ChatBot 데이터 수정
        - 머신러닝 시각화 레이아웃 수정
        - 전월세 전환 계산 2,3 구현( 월세--> 전세, 전환율 계산)
</div>
</details>

<details>
<summary><h4>2023/02/09</h4></summary>
<div markdown="1">

- 개발팀, 데이터팀
    + 오늘 한 것
        - streamlit 배포
        - 시스템 통합
    + 오늘 못한 것
        - 시스템 통합
        - 버그 수정
    + 내일 할 것
        - 시스템 통합
        - 버그 수정
</div>
</details>

<details>
<summary><h4>2023/02/10</h4></summary>
<div markdown="1">

- 개발팀, 데이터팀
    + 오늘 한 것
        - 버그 수정
        - 시스템 통합
        - 마무리
</div>
</details>

</div>
</details>
