# TSTL project

New Run 팀

윤형석, 이주천, 윤재호, 한은기

--- 

<br>

## day 5/10

### 회의 및 수행 사항

#### data labeling

- yolo label 툴로 진행 완료

#### 추가 데이터셋은 아직

- 추가 데이터셋을 오픈소스를 다운받을지, 자이카 조이스틱해서 핸드메이드 데이터셋 만들지 고민

<br>

<br>

#### hyperparameter

pretrained model쓴다면
- learning_rate : 0.001
- scheduler : x

아니면,
- learning_rate : 0.01
- scheduler : step 5 마다 learning rate 갱신

<br>

### 계획 및 진행 예정 사항

분담
- augmentation : 주천
- loss : 재호
- ros : 은기
- aws : 형석

<br>

<br>


## day 5/11

### 회의 및 수행 사항

분담
- augmentation : 주천
- loss : 재호
- ros : 은기
- aws : 형석

-\>

**드라이브가 시급하다고 판단되어 다시 분담**

- lane detection : 은기
- 신호등 color 인식 : 형석
- data_augmentation : 주천
- loss : 재호

4시반까지 data_aug랑 loss merge하고, lane_detection 부분으로 넘어가기

<br>

lane detection의 경우 교차로에서 차선이 없어지거나 이상하게 생기게 탐지가 된다. 그래서 표지판을 인식해서 기울기 값 범위를 지정해서 조향각 제어

객체가 여러개 잡히는 경우 bbox threshold를 정해서 일정 사이즈 이하의 bbox는 무시, 가장 큰 bbox만을 탐지하도록 알고리즘을 짠다.

<br>

### 계획 및 진행 예정 사항

<br>

<br>

## day 5/12

hyperparameter 조정은 5/12일 진행 예정

pytorch -> tensorRT 해서 추론까지 수행해보기
