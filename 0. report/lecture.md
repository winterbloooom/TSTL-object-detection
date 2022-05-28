yolov3 tiny와 model optimization 강의를 타이핑한 자료입니다.

> data labling과 training using AWS는 노션 페이지에,,, 

<br>

---

# Yolov3-tiny

yolov3-tiny는 yolov3보다 light한 모델로 416x416크기의 이미지에 대해 yolov3는 weight가 61,561,429개이지만, tiny에서는 8,686,046개로 총 1/7~1/8배의 사이즈를 가진다. 

weight가 작은만큼 성능은 당연히 yolov3가 좋지만, 속도면에서는 tiny가 더 빠르다.

<br>

<img src="./assets/tiny.png">

헤더가 2개로, output이 2개를 추출한다. 출력의 크기는 입력 사이즈에서 1/32, 1/16 의 shape을 가진다.

<br>

<br>

- **pretrained weight**

https://pjreddie.com/darknet/yolo/

darknet에 대한 pretrained weight를 darknet 사이트에서 다운받을 수 있다.

<br>

<br>

yolov3-tiny의 특징으로는 MaxPool2d + ZeroPad2d를 사용한다.

- maxpool2d

<img src="./assets/maxpool.png">

max값을 추출하려는 size를 정하고, 그 안에서 max값을 뽑아서 출력한다. 아래의 그림은 size가 2일 것이다.

<br>

- zeropad2d

maxpool의 stride(=size)가 홀수일 때 문제가 생길 수 있다. 예를 들어 입력 size가 3x3이라면 stride가 2로 설정하면 에러가 난다. 이 때, zeropadding 처럼 0을 테두리에 추가하여 4x4로 설정하여 에러가 나지 않도록 맞춰준다.

<br>

```cfg
[maxpool]
size=2
stride=1
```

maxpool에 대한 연산을 하기 위해서는 코드에서 더 추가해줘야 한다.

<br>

<br>

- **change num_classes**

num_classes를 변경해주기 위해서는 config 파일에서 3가지를 변경해줘야 한다.

- 맨 위에 classes 변경
- yolo layer의 classes를 변경
- yolo layer이전의 convolutional에 filters(=output channels)을 맞게 변경
    - filters : (box[4] + obj[1] + class_conf[num_class]) * num_anchor = (4 + 1 + 6) * 3 = 33

<br>

<br>

## architecture

residual block을 사용하여 이전의 특징을 연산의 결과와 결합해서 더 좋은 성능을 가진다. conv 8번 layer를 나중에 연산한 특징맵을 upsampling 해서 두 개를 concat해준다. 또한, conv와 batchnorm을 각각 연산하는 것이 아닌 conv + batchnorm를 함께 연산하여 속도를 올린다.



<br>

## code

training arg
- checkpoint : 중간에 학습을 중단하여 **중간단계**에서 다시 시작하고 싶을 때 저장된 pth파일을 불러와 이 weight부터 진행이 된다.
- pretrained : darknet weight를 넣어서 사용하면 된다. 

evaluate
- mode : eval
- 전과 동

<br>

test
- mode : demo
- 전과 동

<br>

torch2onnx
- mode : onnx
- model/yolov3.py 안에 있는 `ONNX_EXPORT = True`로 변경해야 함
- 전과 동 

<br>

<br>

tensorboard
```bash
tensorboard --logdir=./output --port 8888
```

<br>

yolov3-pytorch 코드 흐름 및 
1. cfg param 불러와서 저장
2. gpu check
3. args.mode에 따라 코드
4. train
5. transform, yolodata, dataloader
    - collate_fn : 원래는 1개씩 되어 있는 데이터를 batch 수만큼 묶어서 batch shape을 만들어서 데이터를 만들어주는 것
    - drop_last : 마지막 부분 버릴지
    - shuffle : 데이터를 섞을지, eval에서는 false
6. Darknet53 불러오기
7. pretrain 넣으면 weight를 불러와 model에 저장, 그렇지 않으면 initialize
    - load darknet weights : 파일을 불러오면 바이너리 파일로 되어 있음. 그것을 읽어들이는 부분
8. gpus
    - 1개 사용하면 i = 0, 사용가능한 메모리와 사용할 메모리를 판단해서 적합하지 않으면 돌리지 않음. 이전의 돌렸던 모델도 함께 터지는 것을 방지
    - gpu가 1개면 `cuda:0`으로 입력
    - gpu가 멀티 코어이면, loss 계산은 각 gpu에서 하지만, backprop는 0번째, default gpu에 모여서 연산이 되므로 default device를 설정해줘야 한다.
9. checkpoint 
    - weight 불러오기
10. summary.summary
    - 모델을 요약해주고, 파라미터, layer, 모델의 비정상적인 부분까지 체크해줌
11. summarywriter
    - tensorboard에 데이터를 올리기 위한 장치
12. yolodata
    - 경로를 자신에 맞게 수정해줘야 한다.
    - image는 PIL로 불러옴
    - annotation 파일 불러와서 bbox 받아오기, bbox가 없으면 스킵(gt가 없는 사진을 학습시키면 원하는 타겟이 없으므로 성능이 올라가거나 하지 않는다.)
    - transform
    - batch_idx,cls,x,y,w,h 형태로 생성
    - eval일 때는 0,0,0,0,0으로 bbox 생성 후 return
13. transform
    - augmentation을 할때 조심해야 할 부분은 flip, 현재 데이터에서는 좌우 반전을 하면 좌회전, 우회전 클래스가 섞이게 됨
    - 그래서 flipaug_tstl을 생성, left sign과 right sign 레이블을 함께 변경시켜줌
    - imgaug에서 보면 fliplr_tstl이라는 이름의 augmentation이 존재한다면 그리고 bbox가 있다면, 그러나 일단은 무조건 통과하므로 존재하긴 하나 50%확률로 flip을 시켜주고 있기 때문에, flip이 되어 잇는지를 판단해야 한다. flip이 되었다면 label을 바꿔준다.
    - remove_out_of_image_fraction : 객체의 40%가 이미지 밖으로 나가면 그 이미지는 삭제
    - clip_out_of_images : crop했을 때 bbox가 밖으로 나가는 것을 남은 부분으로 bbox를 제작해줌
14. trainer
    - lr scheduler 을 사용
    - warm up : 학습된 weight가 없을 때, learning rate가 크면 학습이 어렵다. 그래서 warmup을 통해서 처음에는 매우 작은 lr을 주고, 원래의 lr로 올린 다음 schduler를 통해 단계적으로 작아지게 만든다.
    - lr scheduler가 다양하니 다양한 것을 도전해보는 것을 추천
15. run
    - 50 epoch마다 학습 weight 저장
    - 50 epoch마다 eval 돌도록 만듬
    - max batch가 되면 멈춤
16. run iter
    - drawBox를 통해 input 이미지에 bbox 그려보기
    - model 연산, loss 계산, backward, optimizer, scheduler
    - 100번째마다 torch writer, lr/latency/loss 그래프
17. loss
    - 새로운거 쓸거면 써라
    - `#3`은 class를 구분하여 loss를 구하는 것인데, class들을 카테고리로 묶을 수 있는 것들로 묶어서 계산하는 것, 예를 들어 자전거와 보행자를 하나로 묶고, 차, van, truck등을 묶을 수 있을 것이다. 묶어서 loss를 구함
    - loss weight 주고, 다 더해서 loss list에 넣음
18. eval
    - torch.no_grad()
    - non_max_suppression : 동일한 클래스 내에 겹치는 bbox들 중 가장 좋은 bbox만 고름
    - batch_statistics : 전체 prediction을 다 뽑음
    - 각 클래스마다 ap,precision,recall,f1 그래프를 볼 수 있음
19. yolov3
    - maxpool의 경우 2가지로 나눔
        - pad는 다 0
        - stride가 1인 경우 zeropad 수행, 나머지는 그냥 maxpooling
    - x.view 해서 shape 변경, batch, num_anchor * num_attribute, x_height, x_width -> batch, num_anchor, x_height, x_width, num_attrib
    - yolo_grid : yolov3에서는 3개를 다쓰고, tiny에서는 앞에 2개만
20. demo
    - 저장되게끔 만들었는데, drawboxlist를 하면, 이미지 출력되는 것들 다 save해서 바로 볼 수 있게 해놓았다.

<br>

우리가 직접 조정할 만 한 값들
- config : resolution change, batch size
- hyperparameter : optimizer, weight decay of SGD, activation
- augmentation
    - 데이터셋이 별로 없고, 외부 데이터를 사용하더라도 많이 없어서 적절한 것을 사용
- loss를 새로운 것을 정의
- 가까운 object를 잘 찾을 수 있게 설정
    - 내 생각에는 나중에 NMS나 bbox를 필터링할 때 bbox 크기에 대한 threshold를 지정해서 큰 거에 대해서만 추출하도록 만듦

<br>

<br>

---

# export

1. pytorch to darknet
- 자이카에서 지원하는 버전이 구버전이라 pytorch에서 onnx로 바로 변환이 안된다. darknet에서 onnx는 지원하므로 pytorch를 먼저 darknet으로 변환
- AWS 서버에서 진행
2. darknet to onnx
- darknet에서 onnx로 변환
- xycar에서 진행
3. onnx to tensorRT
- onnx는 많은 프레임워크로 포팅할수 있도록 해주는 프레임워크이다.
- 하드웨어 환경에 맞게 tensorRT로 변환
- xycar에서 진행
4. inference test
- xycar에서 진행

## 1. pytorch to darknet

yolov3-pytorch에서 mode를 onnx로 하고, ONNX_EXPORT = True로 설정해야 한다. 이는 tensorRT에는 upsample의 scale_factor를 지원하지 않아서 다르게 구현해놨기 때문이다.

<br>

그래서 먼저

```bash
python main.py --mode onnx --cfg ./config/yolov3-tiny_tstl_416.cfg --pretrained ./config/model_epoch6.pth
```

를 수행하고 나면 weights 파일이 생성될 것이다. 이는 darknet weight파일이다.

<br>

이 파일은 AWS 서버에 있는 파일이므로 옮겨줘야 한다. 옮겨줄 때는 `scp`명령어를 사용하거나 filezilla 프로그램을 설치한다. ftp통신을 통해 aws 서버와 파일을 주고 받는다. 이를 통해 파일을 자이카로 옮겨준다.

<br>

## 2. darknet to onnx

yolov3_onnx_rt repository를 클론해서 가이드대로 실행하면 된다.

먼저 pip버전을 확인해본다.

```bash
pip -V
```

이렇게 했을 때 없다고 뜨거나 3.x 버전이면 2.7을 설치한다.

<br>

### yolov3_to_onnx.py 

1. parser
- cfg : config 파일 경로
- weights : weight 파일 경로
- num_class : 6

2. darknetparser
- config 파일 받아오기

3. output channels 설정
- ( num_class + bbox[4] + obj[1] ) * num_anchor

4. dims
- tiny 이면 1/32, 1/16만 사용
- tiny아니면 1/32, 1/16, 1/8 사용

5. onnx변환 빌더 생성
- darknet weight와 config 파일을 통해 yolov3 model을 빌드
- layer들을 보면서 하나하나 onnx로 포팅
- 하나하나 파보면 공부는 되지만, 이 후 버전에서는 pytorch -> onnx가 바로 가능하기 때문에 공부만 될듯

```bash
sudo python yolov3_to_onnx.py --cfg ./config/yolov3-tiny_tstl_416.cfg --weights ./config/yolov3-tiny_tstl_416.weights --num_class 6
```

이렇게 하면 save되고, .onnx 파일로 생성이 된다.

이 때, need more than 1 value to unpack 에러가 뜨면, cfg file의 형식이 다르다는 것이다. 즉, 확장자까지 포함하는 프로그램을 실행하면 `CRLF`로 되어 있을 것이다. 이를 `LF`로 변환해준다.

<br>

## 3. onnx to tensorRT

yolov3_onnx_rt 레포지토리의 onnx_to_tensorrt.py를 실행한다. onnx를 tensorRT로 변환해주는 코드인데, inference test까지 해주는 코드이다.

1. cfg 파일을 가져와서, 모델의 정보를 가져온다.
2. preprocessYOLO : 입력 이미지의 resize 작업을 해줌
    - postprocessor : nms나 threshold를 통해 최종적인 output을 내기 위한 과정
3. get_engine : 
    - engine : tensorRT를 의미한다.
    - onnx 를 통해 tensorRT로 변환해주는 작업을 수행
    - build engine : network를 생성하고, 모델을 만들기 위한 메모리 수등을 미리 설정해서 tensorRT로 생성
    - do_inference : input이 들어가서 trt(tensorRT) output으로 나온다.
    - postprocessor를 통해 box와 class, score로 결과를 만든다.

```bash
sudo python onnx_to_tensorrt.py --cfg ./config/yolov3-tiny_tstl_416.cfg --onnx ./config/yolov3-tiny_tstl_416.onnx --num_class 6 --input_img ../../xycar_ws/src/test/src/ video1_2.png
```

이를 실행하면 bbox와 클래스 index, score가 추출된다. 그리고 이 결과는 predict.png 파일과 trt 파일로 저장된다.

> label.txt 파일이 잘 되어 있는지 확인

<br>

# ros

yolov3_trt_ros 레포지토리에서 실행하고, trt 파일을 사용하여 

trt_detection.py
- yolo = yolov3_trt() 클래스 지정
- init_node
    - yolov3_trt_ros 라는 노드를 생성
- yolo.detect()
    - ros와 결합해서 이미지가 입력으로 들어오면 바로바로 inference하게 되어 있다.
        - trt engine에 있는 input output 등을 정의
        - preprocessor : image가 들어오면 이미지를 resize해줌
        - input[0].host : image를 입력에 넣어줌
        - do_inference : 입력 이미지를 inference
        - preprocessor : nms를 통해서 우리가 원하는 bbox와 class, score를 얻음
        - publisher : 이를 다음 노드로 전송
            - bbox라는 custom메세지의 array를 만들고, bbox는 xmin, ymin, xmax, ymax, class_id, probability를 담고 있음
            - write_message : inference된 bbox, class, score들을 묶어서 msg 타입에 넣어주고, trt_drive로 publish

trt_drive.py
- BoundingBoxes 메시지 타입을 받게 되고 이를 받으면 이를 통해 처리
- object id에 따라서 left, right, stop, turn 등 조향

