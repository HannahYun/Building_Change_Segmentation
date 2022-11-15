# 국방 AI 경진대회 코드 사용법
- H3J1팀, 김현빈, 김형섭, 윤한나, 정재윤
- 닉네임 : 김면봉이, 나시고래, 여름여름, lastdefiance20
 
 
# 핵심 파일 설명
모든 파일의 경로 및 실행 코드는 압축된 zip파일을 푼 이후, zip파일을 푼 폴더로 들어간 이후의 경로를 상정하고 있습니다.
 
## 파일 구조 및 핵심 파일 설명
```
┖ config
  ┖ predict.yaml - 일반 및 TTA 테스트 실행시 사용하는 arg 파일
  ┖ predict2.yaml - 모델 2개 앙상블 TTA 테스트 실행시 사용하는 arg 파일
  ┖ train.yaml - 학습시 사용하는 arg 파일
┖ models
  ┖ utils.py - 변수에 따라서 segmentation_models_pytorch 라이브러리 모델을 불러오는 코드
┖ modules
  모듈이 구현된 .py 파일들이 내부에 존재
  ┖ datasets.py - 주어진 input data를 반으로 잘라 두 개의 이미지를 얻은 후 각각의 이미지를 90도씩 4번 돌려서 총 4배의 데이터를 얻음
  ┖ earlystoppers.py - 학습이 n번이상 개선되지 않을시 학습을 중단함
  ┖ losses.py - 학습에 사용할 loss 함수들
  ┖ metrics.py - iou, miou 계산 후 plot생성
  ┖ optimizers.py - 학습에 사용한 optimizer함수들
  ┖ recorders.py - 학습 결과 기록해주는 함수
  ┖ scalers.py - 이미지 분포를 scale해주는 함수(CLAHE 추가)
  ┖ schedulers.py - learning rate scheduler들(ExponentialLR 추가)
  ┖ trainer.py - 학습 및 valid를 진행하는 class
  ┖ utils.py - 기타 함수들 (파일 IO 및 로그)
┖ data
  이 폴더 내부에 대회에서 제공한 train.zip와 test.zip을 unzip하여 다음과 같은 경로로 설정한다.
  ┖ train
    ┖ x
    ┖ y
  ┖ test
    ┖ x
┖ results
  이 폴더 내부에는 추후에 모델을 돌리면 모델과 그래프가 저장된다.
  ┖ pred
    테스트 실행 코드 결과 폴더가 생성됨
  ┖ train
    학습 실행 코드 결과 폴더가 생성됨
    ┖ noyangsim_v1
      ┖ model.pth - 첫번째 모델 가중치 파일 (학습된 가중치 파일)
    ┖ noyangsim_d7
      ┖ model.pth - 두번째 모델 가중치 파일 (학습된 가중치 파일)
┖ require.ipynb - 라이브러리 설치 코드
┖ predict.py - 일반 테스트 실행 코드
┖ predict_TTA.py - TTA 테스트 실행 코드
┖ predict_TTA_2.py - 2개 앙상블 TTA 테스트 실행 코드
┖ preprocess.py - 데이터 증강 코드
┖ train.py - 학습 실행 코드
┖ README.md
┖ noyangsim_r4d7_TTA_Ens.zip (추론 결과 제출 파일)
```
 
  - 학습 데이터 경로: `./data/train`
  - Network 내부 encoder 초기 값으로 사용한 공개된 모델 가중치:
    - `https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth`
  - 공개 Pretrained 모델 기반으로 추가 Fine Tuning 학습을 진행한 모델 가중치
    - `./results/train/noyangsim_v1/model.pth`
    - `./results/train/noyangsim_d7/model.pth`
 
  - 이미지 증강 코드: `./preprocess.py`
  - 학습 메인 코드: `./train.py`
  - 테스트 메인 코드: `./predict.py`
  - 테스트 결과 이미지 경로: `./results/pred/{pred.yaml에 입력한 train_serial + 2022~}/mask`
 
## 코드 구조 설명
- baseline으로 주어진 segmentation_models_pytorch를 backend로 사용하여 학습 및 테스트
    - 최종 사용 모델 : segmentation_models_pytorch에서 제공하는 DeepLabV3Plus 모델
    - Data Loader에서 scaler 추가 : clahe
    ```
   ./modules/scalers.py (line 16~17, line 38~58)
   ./configs/train.yaml (line 13)
    ```
    - Trainer에서 scheduler 추가: ExponentialLR
    ```
   ./modules/schedulers.py (line 15~17, line 38~58)
   ./configs/train.yaml (line 47~48)
    ```
 
- **최종 제출 파일 : noyangsim_r4d7_TTA_Ens.zip**
- **학습된 가중치 파일**
- **./results/train/noyangsim_v1/model.pth**
- **./results/train/noyangsim_d7/model.pth**
 
## Code Running Environment
>## run at ubuntu
>**use docker image of paperspace gradient Scratch docker**
>https://hub.docker.com/r/paperspace/gradient-base 
>
> **after then, run require.ipynb to install required library**
>
> **docker tag** \
> pt112-tf29-jax0314-py39-20220803
>
## Machine Environment
> a5000 24GB Vram
 
### docker 컨테이너를 불러온 후 설치된 library들의 상세한 version
- pickle5==0.0.11
- segmentation_models_pytorch==0.3.0
- tabulate==0.9.0
- einops==0.6.0
 
# 학습 과정 준비 방법
- 학습 데이터 경로 설정
```
  ┖ data
      이 폴더 내부에 대회에서 제공한 train.zip와 test.zip을 unzip하여 다음과 같은 경로로 설정한다.
      ┖ train
	    ┖ x
	    ┖ y
      ┖ test
	    ┖ x
```

이후 데이터 증강 코드를 실행한다.
```
python preprocess.py
```
 
# 학습 실행 방법
- 첫 번째 모델 학습을 위해 다음과 같이 train.yaml 파일을 설정한다.
```
# System
gpu_num: 0
seed: 42
debug: False
verbose: False
 
# Train / val
val_size: 0.1
 
# Data Loader
input_width: 448
input_height: 224
scaler: clahe
num_workers: 4
shuffle: True
drop_last: False
 
# Model
architecture: DeepLabV3Plus
encoder: resnext101_32x8d # timm-regnety_016
encoder_weight: imagenet # noisy-student
depth: 6
n_classes: 4
activation: null
 
# Trainer
n_epochs: 100
batch_size: 32
loss: 
  name: MeanCCELoss # MeanCCELoss
  args:
    weight: [1, 1, 1, 1]
metrics: 
  - miou
  - iou1
  - iou2
  - iou3
earlystopping_target: val_miou
earlystopping_patience: 10
optimizer:
  name: AdamW
  args:
    lr: 5.0e-04
    weight_decay: 2.0e-02
scheduler:
  name: ExponentialLR
  args:
    gamma: 0.96
 
# Logging
plot:
  - loss
  - miou
  - iou1
  - iou2
  - iou3
```
 
이후 학습 코드를 실행한다.
```
python train.py
```
 
실행 이후, 20번째 epoch에서 ctrl+c를 통해 학습을 중단했다 (시간관계상)
실행 이후에 results/train 내부 생성된 폴더의 이름을 기억한다.
 
- 두 번째 모델 학습을 위해 다음과 같이 train.yaml 파일을 설정한다.
```
# System
gpu_num: 0
seed: 42
debug: False
verbose: False
 
# Train / val
val_size: 0.1
 
# Data Loader
input_width: 448
input_height: 224
scaler: clahe
num_workers: 4
shuffle: True
drop_last: False
 
# Model
architecture: DeepLabV3Plus
encoder: resnext101_32x8d # timm-regnety_016
encoder_weight: imagenet # noisy-student
depth: 7
n_classes: 4
activation: null
 
# Trainer
n_epochs: 100
batch_size: 32
loss: 
  name: MeanCCELoss # MeanCCELoss
  args:
    weight: [1, 1, 1, 1]
metrics: 
  - miou
  - iou1
  - iou2
  - iou3
earlystopping_target: val_miou
earlystopping_patience: 10
optimizer:
  name: AdamW
  args:
    lr: 5.0e-04
    weight_decay: 2.0e-02
scheduler:
  name: ExponentialLR
  args:
    gamma: 0.96
 
# Logging
plot:
  - loss
  - miou
  - iou1
  - iou2
  - iou3
```
 
이후 학습 코드를 실행한다.
```
python train.py
```
 
실행 이후, 20번째 epoch에서 ctrl+c를 통해 학습을 중단했다 (시간관계상)
실행 이후에 results/train 내부 생성된 폴더의 이름을 기억한다.
 
# 테스트 실행 방법
현재까지 우리는 train 내부 생성된 두개의 폴더이름을 기억하고 있다.
 
그 폴더이름을 predict_2.yaml 내부 시리얼 변수에 적어준다.

- train_serial: "첫번째 학습으로 생성된 폴더이름"
- train_serial_2: "두번째 학습으로 생성된 폴더이름"

시리얼 변수를 바꿔줌으로써 모델 2개를 사용한 앙상블 및 TTA 테스트가 준비되었다.
 
- 테스트 코드 실행
```
python predict_TTA_2.py
```
 
이후 추론 결과는 ./results/pred/ 경로 내부에 생성되어있으며, 생성된 폴더 내부 mask 폴더 안에 추론된 mask값이 저장되어있다. 이를 최종적으로 zip하여 제출하면 된다.

