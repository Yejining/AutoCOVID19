# COVID19 SOUTH KOREA DATASET

### Structure

``` bash
data
│   README.md    
│
└───raw
│   └───github
│   │   │   Case.csv
│   │   │   Patient_info.csv
│   │   │   Patient_Route.csv
│   │  
│   └───seoul
│   |   │   ...
|   |
|   └───status
│   |   │   corona_status.html
|   |
|   └───...
|
└───extracted
    │   MergedRoute.csv
    │   modified_route.csv
    │   corona_status.csv
    │   ...
```



### 원본 데이터셋

1. github에서 가져온 데이터

   - `data\raw\github\` 하위 파일들
   - 연구 초기에 사용한 데이터
   - [출처](https://www.kaggle.com/kimjihoo/coronavirusdataset)

2. 2020년 5월 ~ 2020년 8월 데이터

   - `data\raw\seoul\` 하위 파일들
   - 5월부터 8월까지 정리된 데이터

3. 2020년 8월~ 데이터

   - `data\raw\status\corona_status.html`
     - 확진자 환자 번호, 확진일, 거주지, 여행력, 접촉력, 퇴원 현황 파싱
     - [출처](https://www.seoul.go.kr/coronaV/coronaStatus.do#status_page_top)

   - `data\raw\` 위에서 언급한 폴더/파일 제외한 모든 폴더/파일
     - 8월 이후 각 지자체에서 파싱한 동선 데이터(html, png, txt 등)



#### 전처리 데이터셋

1. github에서 가져온 데이터
   - `data\extracted\MergedRoute.csv`에 저장
2. 2020년 5월 ~ 2020년 8월 데이터
   - `data\extracted\modified_route.csv`에 저장
3. 2020년 8월~ 데이터
   - `data\extracted\corona_status.csv`
     - `data\extracted\corona_status.html` 파싱한 데이터
   - `data\extracted\`에서 언급한 파일 제외한 모든 파일
     - 8월 이후 각 지자체에서 가져온 데이터 파싱한 파일 csv 형태로 변환한 것