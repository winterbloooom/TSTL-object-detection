# 초기 모델

- boxloss
<img src="./assets/boxloss.png">

- clsloss
<img src="./assets/clsloss.png">

- latency
<img src="./assets/latency.png">

- learning rate
<img src="./assets/lr.png">

- objectness loss
<img src="./assets/objloss.png">

- total loss
<img src="./assets/totalloss.png">

- evaluation
<img src="./assets/evaluation.png">

<br>

사용한 알고리즘
- optimizer : SGD, momentum 0.9, decay 0.0005
- learning rate : 0.001
- loss function : bcelogloss
- lr_scheduler : MultiStepLR, gamma - 0.5, milestones : [10000,20000,30000]
- lcls : 0.5
- lobj : 1.0
- lbox : 0.05

```bash
epoch 0 / iter 96 lr 0.00010 , loss 1.64410 latency 0.04697
epoch 6 / iter 1073 lr 0.00100 , loss 1.00232 latency 0.03125
epoch 19 / iter 3523 lr 0.00100 , loss 0.25401 latency 0.02698
epoch 28 / iter 5137 lr 0.00100 , loss 0.25522 latency 0.03165
epoch 55 / iter 9952 lr 0.00100 , loss 0.34121 latency 0.03124
```

bcelogloss는 2개의 클래스를 가질 때 사용하는 loss이다. beclogloss를 사용하기 위해서는 output shape이 [num_batch, 1]로 나와야 하고, 마지막 output 값들이 0~1값으로 나와야 하므로 마지막에 activation function을 추가해줘야 한다.

<br>

<br>

## fitting loss weight

- lcls : 0.5
- lobj : 1.0
- lbox : 0.05

```bash
epoch 0 / iter 96 lr 0.00010 , loss 1.64410 latency 0.04697
epoch 6 / iter 1073 lr 0.00100 , loss 1.00232 latency 0.03125
epoch 28 / iter 5021 lr 0.00100 , loss 0.65669 latency 0.02645
epoch 55 / iter 9952 lr 0.00100 , loss 0.34121 latency 0.03124
```

<br>

- **lcls : 0.05**
- **lobj : 1.0**
- **lbox : 0.5**

```bash
epoch 0 / iter 1 lr 0.00000 , loss 1.62597 latency 0.01804
epoch 6 / iter 1076 lr 0.00100 , loss 0.84359 latency 0.03100
epoch 28 / iter 5044 lr 0.00100 , loss 0.50940 latency 0.03125
epoch 50 / iter 9109 lr 0.00100 , loss 0.05907 latency 0.01700
```

성능이 확실히 좋아졌음

<br>

- lcls : 0.25
- lobj : 1.0
- lbox : 0.25

```bash
epoch 0 / iter 1 lr 0.00000 , loss 1.53861 latency 0.01560
epoch 6 / iter 1069 lr 0.00100 , loss 0.95907 latency 0.03126
```

x

<br>

- lcls : 0.05
- lobj : 0.5
- lbox : 0.5

```bash
epoch 0 / iter 1 lr 0.00000 , loss 1.35323 latency 0.32276
epoch 6 / iter 1078 lr 0.00100 , loss 0.97386 latency 0.03800
epoch 19 / iter 3511 lr 0.00100 , loss 0.57411 latency 0.04000
```

<br>

## optimizer

원래는 SGD

Adam으로 변경

- Adam
- lr : 0.001
- lcls : 0.05
- lobj : 1.0
- lbox : 0.5

```bash
epoch 0 / iter 1 lr 0.00000 , loss 1.67407 latency 0.34670
epoch 6 / iter 1073 lr 0.00100 , loss 0.55838 latency 0.03125
epoch 6 / iter 1169 lr 0.00100 , loss 0.49542 latency 0.01562
...
epoch 29 / iter 5402 lr 0.00100 , loss 0.73019 latency 0.02278
epoch 29 / iter 5500 lr 0.00100 , loss 0.63979 latency 0.03326
epoch 30 / iter 5589 lr 0.00100 , loss 0.63169 latency 0.03071
```

loss가 수렴이 안되고 있음

<br>

- Adam
- lr : 0.0001
- 전과 동

```bash
epoch 0 / iter 1 lr 0.00000 , loss 1.67407 latency 0.34670
epoch 0 / iter 1 lr 0.00000 , loss 1.67407 latency 0.34670
...
epoch 29 / iter 5012 lr 0.00000 , loss 1.47407 latency 0.34670
epoch 30 / iter 5505 lr 0.00000 , loss 1.27407 latency 0.34670
```

아예 최저점에 접근하는 게 거의 안되고 있음

<br>

맨 처음에 가장 잘 나왔던 셋으로 진행

- optimizer : SGD, momentum 0.9, decay 0.0005
- learning rate : 0.001
- loss function : bcelogloss
- lr_scheduler : MultiStepLR, gamma - 0.5, milestones : [10000,20000,30000]
- lcls : 0.5
- lobj : 1.0
- lbox : 0.05

```bash
epoch 39 / iter 7274 lr 0.00100 , loss 0.78654 latency 0.02363
epoch 39 / iter 7372 lr 0.00100 , loss 0.68753 latency 0.03656
epoch 40 / iter 7459 lr 0.00100 , loss 0.68740 latency 0.02344
epoch 40 / iter 7558 lr 0.00100 , loss 0.69155 latency 0.02375
epoch 41 / iter 7647 lr 0.00100 , loss 0.61295 latency 0.01835
epoch 41 / iter 7744 lr 0.00100 , loss 0.84163 latency 0.01979
epoch 42 / iter 7834 lr 0.00100 , loss 0.55604 latency 0.02935
```

그러나 갑자기 동일하게 0.5에서 안내려가기 시작

<br>

augmentation이 잘못되었거나 충분한 iteration을 하지 않았기 떄문인듯 함.

<br>

## scheduler

