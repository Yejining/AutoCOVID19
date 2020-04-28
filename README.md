# AutoCOVID19

### 레퍼런스

["Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction"](https://arxiv.org/abs/1610.00081)



### 기존 프로젝트

- [데이터셋](https://github.com/liweiowl/ST-ResNet20190702)
- [코드1](https://github.com/snehasinghania/STResNet)
  - 버전 낮아서 실행 못함
- [코드2](https://github.com/BruceBinBoxing/ST-ResNet-Pytorch/)
  - BikeNYC까지 구현되어 있고, TaxiBJ까지 실행되는 코드는 없음



### 실행 방법

1. 데이터셋

   1. [OneDrive](https://1drv.ms/f/s!Akh6N7xv3uVmhOhCtwaiDRy5oDVIug)에서 BikeNYC 데이터 다운 받고 `data/BikeNYC/`에 저장
   2. [OneDrive](https://1drv.ms/f/s!Akh6N7xv3uVmhOhCtwaiDRy5oDVIug)에서 TaxiBJ 데이터 다운 받고 `data/TaxiBJ/`에 저장

2. BikeNYC 실행

   1. Prerequisites

      ```
      Python 3.6
      Pytorch 1.0
      ```

   2. ```
      python projects/BruceBinBoxing-ST-ResNet-Pytorch/train_bikenyc_stresnet.py
      ```

      

### COVID19

1. 데이터셋
   1. `covid19/README.md`의 링크에서 데이터셋 다운로드