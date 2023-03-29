# YOLOv3-pytorch

---

> <u>**해당 레포지토리는 이다민 강사님([2damin](https://github.com/2damin))의 레포지토리 [yolo3-pytorch](https://github.com/2damin/yolov3-pytorch)를 수정해 사용하고 있음을 밝힙니다.**</u>

## 레포지토리 구성
```bash
│
└─ dataloader/             # 이미지 데이터를 읽어오기
   └─ data_transfrom.py
         # 데이터 transform을 정의함(image augmentation)
   └─ yolodata.py
         # 저장된 파일(데이터셋)을 읽어옴
└─ demo/ # 모델을 실행(테스트)하는 모드
   └─ demo.py
         # demo 모드를 위한 모델 실행 클래스/함수 정의
└─ eval/
   └─ evaluator.py
         # 평가(점수 산출)를 위한 클래스 및 함수 정의
└─ model/
   └─ load_model.py
   └─ loss.py
         # 손실함수(교차 엔트로피, 이진 교차 엔트로피) 정의
   └─ yolov3.py
         # YOLOv3의 신경망인 DarkNet53을 정의. pretrained weight 처리.
└─ output/  # onnx, weights 결과 저장
└─ train/   # 모델 학습
   └─ loss.py
         # YOLO 레이어에서 산출하는 값(class, objectiveness, bbox)에 대한 Loss를 계산
   └─ scheduler.py
         # scheduler 정의
   └─ trainer.py
         # 학습 과정 정의
└─ util/
   └─ tools.py
         # 비최대억제, bbox 정보 변환, iou 계산 등 함수 정의
└─ common.py
└─ data_processing.py
└─ main.py        # 모델을 사용할 모드 설정 및 수행 등
└─ onnx_infer.py
└─ onnx2rt.py
└─ requirements.txt  # 해당 패키지를 사용하기 위한 사전 설치 목록
└─ yolov3-tiny.weights  # pretrained weights
└─ yolov3-tiny_tstl416.cfg  # 신경망의 환경설정(레이어 정보 등) 파일
```

---

single stage object detection Yolov3.

This is made with Pytorch.


<img src=https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-24_at_12.52.19_PM_awcwYBa.png width=416>

----------------------------

## Install

### Windows

#### Use Anaconda

1. Download Anaconda : https://www.anaconda.com/products/individual#windows
2. conda create --name ${environment_name} python=3.8
3. activate ${environment_name}
4. git clone https://github.com/2damin/yolov3-pytorch.git


### Linux

#### Use docker

I recommend Nvidia NGC docker image. [link](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

1. docker pull nvcr.io/nvidia/pytorch:xx.xx-py3
2. docker run --gpus all -it --rm -v local_dir:container_dir -p 8888:8888 nvcr.io/nvidia/pytorch:xx.xx-py3
   1. check "nvidia-smi"
   2. check "nvcc --version"
3. git clone https://github.com/2damin/yolov3-pytorch.git


## Dependency

pip install -r requirements.txt
```
python >= 3.6

Numpy

torch >= 1.9

torchvision >= 0.10

tensorboard

tensorboardX

torchsummary

pynvml

imgaug

onnx

onnxruntime
```

-------------------

## Run

If training,

```{r, engine='bash', count_lines}
(single gpu) python main.py --mode train --cfg ./yolov3.cfg --gpus 0 --checkpoint ${saved_checkpoint_path}

(multi gpu) python main.py --mode train --cfg ./yolov3.cfg --gpus 0 1 2 3 --checkpoint ${saved_checkpoint_path}
```

If evaluate,

```{r, engine='bash', count_lines}
python main.py --mode eval --cfg ./yolov3.cfg --gpus 0 --checkpoint ${saved_checkpoint_path}
```

If test,

```{r, engine='bash', count_lines}
python main.py --mode demo --cfg ./yolov3.cfg --gpus 0 --checkpoint ${saved_checkpoint_path}
```

If converting torch to onnx,

target tensorrt version > 7
```{r, engine='bash', count_lines}
python main.py --mode onnx --cfg ./cfg/yolov3.cfg --gpus 0 --checkpoint ${saved_checkpoint_path}
```

target tensorrt version is 5.x

1. **ONNX_EXPORT = True** in 'model/yolov3.py'
   
   tensorrt(v5.x) is not support upsample scale factor, so you have to change upsample layer not using scale factor.

```{r, engine='bash', count_lines}
python main.py --mode onnx --cfg ./cfg/yolov3.cfg --gpus 0 --checkpoint ${saved_checkpoint_path}
```

### option

--mode : train/eval/demo.

--cfg : the path of model.cfg.

--gpu : if you use GPU, set 1. If you use CPU, set 0.

--checkpoint (optional) : the path of saved model checkpoint. Use it when you want to load the previous train, or you want to test(evaluate) the model.

--pretrained (optional) : the path of darknet pretrained weights. Use it when you want to fine-tuning the model.



## Visualize training graph

Using Tensorboard,

```{r, engine='bash', count_lines}
tensorboard --logdir=./output --port 8888
```

-------------------------

# Reference

[YOLOv3 paper](https://arxiv.org/abs/1804.02767)

[KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)


# error

- if libgl.so error when cv2
```
apt-get update
apt-get install libgl1-mesa-glx
```