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
│   │   │   ...
│   │
│   └───status
│   │   │   corona_status.html
│   │
│   └───...
│
└───checklist
│   │   names.csv
│   │   type.csv
│   │   kor_type.csv
│   │   infection_case.csv
│
└───extracted
    │   MergedRoute.csv
    │   modified_route.csv
    │   corona_status.csv
    │   ...
    │   merged_route_check_01.csv
    │   merged_route_check_02.csv
    │   merged_route_check_03_01.csv
    │   merged_route_check_03_02.csv
    │   merged_route_check_04_01.csv
    │   merged_route_check_04_02.csv
    │   merged_route_check_04_03.csv
    │   merged_route_final.csv
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
3. 2020년 8월~ 각 구청별 데이터
   - `data\extracted\corona_status.csv`
     - `data\extracted\corona_status.html` 파싱한 데이터
   - `data\extracted\`에서 언급한 파일 제외한 모든 파일
     - 8월 이후 각 지자체에서 가져온 데이터 파싱한 파일 csv 형태로 변환한 것
4. 2020 8월~ 전처리 데이터
   - 모든 지자체의 csv 파일 합해서 코딩으로 전처리한 파일이 `data\extracted\merged_route_check_01.csv`
   - `data\extracted\merged_route_check_01.csv`에서 코딩 + 수작업으로 전처리한 파일이 `data\extracted\merged_route_check_02.csv`
   - `merged_route_check_02.csv`에서 환자 정보(id) 없는 경우 `merged_route_check_03_01.csv`로, 환자 정보가 있는 경우에는 `merged_route_check_03_02.csv`로 따로 분류해놓음
   - `merged_route_check_03_02.csv`에서 환자 일련번호 알아내 감염경로 추출, `merged_route_check_04_01.csv`에 저장
   - `merged_route_check_04_01.csv`에 저장된 데이터에서 geocoder api 이용해 위/경도 추출하고 `merged_route_check_04_02.csv`, `merged_route_check_04_03.csv`에 저장
5. 1~4에서 작업한 데이터
   - 모두 통합해서 `merged_route_final.csv`에 저장



#### 전처리 시 사용한 데이터

- 정규화하기 위해 사용한 파일들 `data\checklist\~`

