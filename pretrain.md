# pretrained weight를 사용하는 것에 따른 코드와 설명

> 다민 강사님의 model_epoch6.pth를 사용

pretrained weight를 사용하지 않고 돌릴 때는 loss가 출력이 된다.

```bash
python main.py --mode train --cfg ./config/yolov3-tiny_tstl_416.cfg --gpus 0  
```

```python
print(self.bcellogloss(pout[...,4], tobj).item(), self.mseloss(pout[...,4], tobj).item())
```

```markdown
1.0560895204544067 578.477783203125
```

---

<br>

그러나 pretrained weight를 사용했을 경우 loss가 Nan이 출력되는 것을 확인

```bash
python main.py --mode train --cfg ./config/yolov3-tiny_tstl_416.cfg --gpus 0 --pretrained ./config/model_epoch6.pth
```

```python
print(self.bcellogloss(pout[...,4], tobj).item(), self.mseloss(pout[...,4], tobj).item())
```

```markdown
learing rate :  0.01
nan nan
```

<br>

Nan값이 나오는 원인
1. learning rate 가 너무 높다.
2. input data에 Nan이 껴있다.
3. output size와 데이터가 불일치
4. log(0)이 어디선가 수행
5. 다른 optimizer를 사용

## 1. learning rate 조정

```bash
learing rate :  0.001
nan nan
```

```bash
learing rate :  0.0001
nan nan
```

```bash
learing rate :  1e-05
nan nan
```

이건 아닌거 같다.

## 다른 optimizer

- Adam

```python
self.optimizer = optim.Adam(model.parameters(), lr=hparam['lr'], weight_decay = hparam['decay'])
```

```bash
optimizer :  Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.001
    weight_decay: 0.0005
)
nan nan
```

<br>

- SGD

```python
self.optimizer = optim.SGD(model.parameters(), lr=hparam['lr'], momentum=hparam['momentum'], weight_decay=hparam['decay'])
```

```bash
optimizer :  SGD (
Parameter Group 0
    dampening: 0
    lr: 0.001
    momentum: 0.9
    nesterov: False
    weight_decay: 0.0005
)
nan nan
```

동일하게 해결 되지 않음

```python
self.optimizer = optim.NAdam(model.parameters(), lr=hparam['lr'], weight_decay = hparam['decay'])
```

```bash
optimizer :  NAdam (
Parameter Group 0
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.001
    momentum_decay: 0.004
    weight_decay: 0.0005
)
nan nan
```

<br>

혹시 scheduler 때문인가?

```python
#self.lr_scheduler.step(self.iter)
```

```bash
optimizer :  SGD (
Parameter Group 0
    dampening: 0
    lr: 0.001
    momentum: 0.9
    nesterov: False
    weight_decay: 0.0005
)
nan nan
```

scheduler 때문도 아님.

<br>

찾아냈다.

```python
            input_img = input_img.to(self.device, non_blocking=True)
            print("input_img : ", input_img)
            
            output = self.model(input_img)
            print(output)

# ================================= #

input_img :  tensor([[[[0.7804, 0.7804, 0.7804,  ..., 0.6667, 0.6667, 0.6667],
          [0.7804, 0.7804, 0.7804,  ..., 0.6667, 0.6667, 0.6667],
          [0.7804, 0.7804, 0.7804,  ..., 0.6667, 0.6667, 0.6667],
          ...,
          [0.5255, 0.5255, 0.5255,  ..., 0.3961, 0.4039, 0.4118],
          [0.5255, 0.5255, 0.5255,  ..., 0.3765, 0.3843, 0.3922],
          [0.5333, 0.5333, 0.5294,  ..., 0.3608, 0.3647, 0.3765]],

output :  [tensor([[[[[nan, nan, nan,  ..., nan, nan, nan],
          [[nan, nan, nan,  ..., nan, nan, nan],
           [nan, nan, nan,  ..., nan, nan, nan],
           ...,
           [nan, nan, nan,  ..., nan, nan, nan],
           [nan, nan, nan,  ..., nan, nan, nan],
           [nan, nan, nan,  ..., nan, nan, nan]]]]], device='cuda:0',
       grad_fn=<CloneBackward0>)]
nan nan
```

```python
    class DarkNet53(nn.Module):
        def forward(self, x):
            ...
            print(idx, " : ", x[:,:,:,0])

# ================== #

0  :  tensor([[[nan, nan, nan,  ..., nan, nan, nan],
         [nan, nan, nan,  ..., nan, nan, nan],
         [nan, nan, nan,  ..., nan, nan, nan],
         ...,
         [nan, nan, nan,  ..., nan, nan, nan],
         [nan, nan, nan,  ..., nan, nan, nan],
         [nan, nan, nan,  ..., nan, nan, nan]],

        [[nan, nan, nan,  ..., nan, nan, nan],
         [nan, nan, nan,  ..., nan, nan, nan],
```

애초에 입력부터 nan으로 들어간다.

<br>

---

---

애초에 pretrained 는 weights 파일이고, checkpoint가 .pth or .pt인데 내가 pretrained 에 .pth를 넣어서 이상한 듯하다.

checkpoint에 .pt를 넣으니 안맞다.

위의 darknet 사이트에서 weight를 다운받아서 `--pretrained ./yolov3-tiny.weights`를 하면 구동이 된다.

