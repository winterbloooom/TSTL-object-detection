# Traffic Sign & Traffic Light Project

- **개요**: 다양한 교통표지(신호등, 횡단보도, 좌/우회전, 정지)를 딥러닝 모델을 통해 실시간으로 인지하고, 해당 표지 지시대로 미니카를 주행하는 프로젝트.
- **일정**: 2022.05.09. - 13.

> K-Digital Training 프로그래머스 자율주행 데브코스 3기의 프로젝트입니다.

|미션 주행 트랙|트랙 주행 장면|
|:---:|:---:|
|<img src="https://user-images.githubusercontent.com/69252153/228587412-b64ac5f4-1a12-4b41-8bb6-5d0779dd26fc.png" style="width:100%">|<img src="https://user-images.githubusercontent.com/69252153/228587721-995ebdfd-08ad-4876-a116-3dde68fdfe34.png" style="width:50%">|

|표지 인식 결과|
|:---:|
|<img src="https://user-images.githubusercontent.com/69252153/228588397-c6d941c3-d9f5-4a94-98d1-0c152598dcdb.png">|


## 주요 수행 내용

🪄 **데이터 라벨링부터 모델 학습, 주행까지 전 과정을 경험**
  - 약 960장의 이미지 데이터 라벨링 & YOLOv3 모델 학습
  - 모델 최적화: Jetson Nano에서 실시간 모델 구동을 위하여 ONNX, TensorRT 변환 과정을 거침


🚥 **교통 표지에 따른 주행 기능 구현**
  - 인식한 교통 표지에 따라 차선 위를 주행할 수 있도록 ROS 프로그래밍
  - 좌/우회전 구간 진입 시, 표지를 벗어나면 직진하는 오류 발생. 추후 코드 리뷰를 통해 오류를 수정함


## Team

|[윤형석](https://github.com/idra79haza)|[이주천](https://github.com/LeeJuCheon)|[윤재호](https://github.com/dkssud8150)|[한은기](https://github.com/winterbloooom)|
|:---:|:---:|:---:|:---:|
|<img src="https://avatars.githubusercontent.com/u/37795618?v=4" alt="img" style="zoom:20%"/>|<img src="https://avatars.githubusercontent.com/u/8746262?&v=4" alt="img" style="zoom:20%"/>|<img src="https://avatars.githubusercontent.com/u/33013780?&v=4" alt="img" style="zoom:20%"/>|<img src="https://avatars.githubusercontent.com/u/69252153?v=4" alt="img" style="zoom:20%"/>|



## Directories
- `1-yolov3-pytorch` : 표지를 인식하기 위한 YOLOv3 모델
- `2-yolov3_onnx_rt` : Jetson에서 모델을 구동하기 위한 최적화. ONNX, Tensor RT 사용
- `3-yolov3_trt_ros` : 표지를 인식하고, 인식한 표지의 지시에 따라 트랙 위를 주행(`trt_drive.py`)하는 코드
