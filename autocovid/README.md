## 전처리 코드

### Dataset structure

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



### Structure

``` bash
src
│   README.md    
│
└───preprocess
│   │   api.py
│   │   parser.py
│   
└───jupyter notebook
    │   1. github dataset preprocessing.ipynb
    │   2-1. Parsing Route.ipynb
    │   2-2. Merge Info.ipynb
    │   2-3. Geo.ipynb
    │   2-4. etc.ipynb
    │   3-1. corona status.ipynb
    │   3-2. 구 ㄱ.ipynb
    │   3-3. 구 ㄴ-ㅁ.ipynb
    │   3-4. 구 ㅅ.ipynb
    │   3-5. 구 ㅇ-ㅈ.ipynb
    │   4-1. merge dataset from august.ipynb
    │   4-2. extract infos using other file or api.ipynb
    │   4-3. merge all dataset.ipynb
    │   9. ConvertingCoordToImage.ipynb
```



1. github에서 가져온 데이터
   - `src\jupyter notebook\1. github dataset preprocessing.ipynb`에서 처리
   - `data\raw\github\` 하위 폴더에서 필요한 데이터 종합해 `data\extracted\MergedRoute.csv`에 결과 저장
2. 2020년 5월 ~ 2020년 8월 데이터
   - `src\jupyter notebook\2~`에서 처리
   - `data\raw\seoul\` 하위 파일들 처리해 `data\extracted\modified_route.csv`에 결과 저장
3. 2020년 8월~ 각 구청별 데이터 파싱
   - `src\preprocess\parser.py`와 `src\jupyter notebook\3~`에서 처리
   - `data\raw\seoul` 하위 파일들 처리해 `src\extracted\` 하위에 지역명으로 결과 저장
   - `data\raw\status\corona_status.html`에서 환자 정보 가져와 `data\extracted\corona_status.csv`에 결과 저장
4. 2020년 8월~ 전처리 데이터
   - 전처리한 데이터 각 `data\extracted\merged_route_check~.csv` 형식으로 저장
   - 전처리는 코딩+수작업으로 진행, 코드는 `src\jupyter notebook\4~`에서 처리
5. 1~4에서 작업한 데이터
   - 모두 통합해서 `merged_route_final.csv`에 저장
   - `4-3. merge all dataset.ipynb`에서 처리



## COVID19 확산 예측

### Structure

``` bash
AutoCOVID19
│   main.py
│   worker_main.py
│   worker_test.py
│
└───src
    │   README.md
    │   Arguments.py
    │   Cases.py
    │   constant.py
    │   Dataset.py
    │   Model.py
    │   process.py
    │   RouteConverter.py
    │   test.py
    │
    └───preprocess
    │   │   api.py
    │   │   parser.py
    │
    └───automl
        └───advisor
        │   │   advisor_main.py
        │   │   advisor_util.py
        │   
        └───hpbandster
            │   worker.py
```



- automl에서 advisor는 폐기한 듯. 잘 기억 안 남.
- automl 실행 안 할 때는 main에서 하고 automl 할 때 worker_main.py 쓰는 듯. 확인 필요함



##  결과 저장

