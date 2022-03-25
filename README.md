## 목차

## 1. 프로젝트 개요

- **대회 정보**
    - 미국의 데이터 Competition Plotform ‘DRIVENDATA’에서 진행 된 Competition
    - Funded by NASA
    
    [Run-way Functions: Predict Reconfigurations at US Airports (Open Arena)](https://www.drivendata.org/competitions/89/competition-nasa-airport-configuration/)
    
- **프로젝트 목적**
    - US 10개 항공 로 미래에 어떤 활주로가 활성화 될지 예측한다
- **프로젝트 설명**
    - 공항의 활주로는 모두 항상 사용되지 않는다. 기상, 시간, 항공량에 따라 활성화 되는 활주로가 다르고 이를 예측하는 것은 효율적인 공항 교통 통제에 있어서 매우 중요하다. 따라서 과거의 데이터들을 이용 해 미래에 어떤 활주로가 활성화 될지 예측하는 모델을 개발한다.
- **데이터**
    - 미국 10개 공항의 공항 별 이착륙 정보, 활주로 활성화 정보, 기상 데이터

## 2. 전처리

- 제공 된 데이터 중 사용 된 것은 활주로 활성화 정보, 기상 데이터가 주로 사용 됨

### 2.1 Configuration

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fb16d530-6930-4e4b-951a-09df49fc9e22/Untitled.png)

- Runway는 크게 Departure 와 Arrival 두종류로 나뉘어 사용 되며 시간, 기상, 항공량에 따라 조합이 구성 됨
- 위 Table의 airport_config 컬럼이 해당 시간대에 활성화 된 runway의 조합을 뜻하며 D_로 시작하는 것들은 이륙에 사용 된 활주로, A로 시작하는 것들은 착륙에 사용 된 활주로를 뜻 함

**시간 간격 통일** 

- 해당 데이터들은 컴퓨터에 자동으로 기록 된 Log 성격의 데이터이기 때문에 기록 시간의 단위가 모두 다름
- 일정한 간격의 시간 단위로 자르기 위해서 30분 간격으로 통일 함

**One- hot Encoding** 

- 이 모델은 일종의 여러개의 Class(Configuration 경우의 수)를 분류하는 Multi class Classification
- 따라서 활성화 횟수가 10회 이하인 것들은 other로 묶고 나머지를 각각 class로 one hot encodling 함

### 2.2 기상 데이터

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/999c55b5-55a6-45d2-bdb9-f4ab8d3049dc/Untitled.png)

- 기상 데이터에는 기록 시간, 예측 시간 그리고 온도, 풍향,풍속,등의 데이터가 존재 함

**Feature** 

- 변수로 풍향과 풍속만 사용 됨
- 비행기가 이착륙시 충분한 양력과 안정성을 위해 맞바람이 불어야 함. 즉 풍향은 다른 어떠한 변수 보다도 활주로 사용에 큰 영향을 끼치기 때문에 변수로 사용 됨

**Time** 

- 이 데이터 역시 Configuration과 마찬가지로 Log 성격의 데이터 이기 때문에 시간 간격이 모두 다름
- 이를 일정한 간격을 맞춰 모델에 input 될 때 특정 범위만 사용될 수 있도록 함

## 3. 모델 설명

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b7eddc14-5c3e-4e67-a44c-e5b98a5b555b/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d531d94f-c4ae-4b81-befe-fb31ab47f288/Untitled.png)

- 기본적인 모델 컨셉은 기준 시간으로 부터 과거 12시간의 데이터를 참조 하여 미래 6시간의 데이터를 예측함
- 과거 12시간 데이터 - 미래 6시간 데이터 그리고 lamp데이터를 모두 연결해주는 key 로 timestamp를 사용 함

### 3.1 **모델 구성**

- 모델은 크게 두개로 나뉨
    - Configuration 데이터를 처리하는 LSTM 네트워크
    - 기상 데이터를 처리하는 임베딩 레이어
- 분류 모델이므로 출력 함수를 softmax를 사용 했으며 loss 로는 categorical crossentropy를 사용 함

**Post process** 

- softmax는 출력 값을 각 class의 확률을 의미 함
- 하지만 submission 에서 각 class의 확률 합이 1이 되도록 요구하기 때문에 모수를 나눔으로써 sub이 1이 되도록 만듬

**10개 공항** 

- 10개 공항의 각 configuration 확률을 모두 각각 예측 해야 함
- 10개 공항을 한번에 학습 하는 것은 수 많은 경우의 수를 유발하기 때문에 정확도가 낮다고 판단 10개 공항 각각 모델을 학습 시키는 것으로 진행 함
- 모델 네트워크는 동일하게 가져가되 데이터만 바꾸어 10개 모델을 만듬

### 3.2. Optimizer

- Optimizer로 Tensorflow Addons API의 **Rectifier Adam**를 사용 함
    - Total steps : 10000
    - warmup_proportion=0.1
    - min_lr=0.00001
    - Adam, SGD, AdamW를 모두 실험을 해 보았지만 RAdam이 가장 좋은 성능을 보여 줌
- Learning rate 조절을 위한 Scheduler로 **Exponential Decay**를 사용
    - Initial Learning rate : 0.001
    - Decay steps : 100000
    - Decay rate : 0.96
    - Stair case : True

## 4. 결과

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3c1c6aae-ca73-45cc-a61e-fb962191be75/Untitled.png)

<aside>
💡 Current Rank : 2등

</aside>
