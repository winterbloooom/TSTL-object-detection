# yolov3(Darknet) -> ONNX -> TensorRT in Jetson TX2

This repo converting yolov3 darknet model to TensorRT model in Jetson TX2 platform.

If you want to convert yolov3 pytorch model, need to convert model from pytorch to DarkNet. Check my [yolov3-pytorch repo](https://github.com/2damin/yolov3-pytorch)

## ENV INFO
ONNX == 1.4.0

TensorRT == 5.1.6.1

Jetson TX2 jetpack == 4.2.3

- CUDA == 10.0.326

- OpenCV == 3.3.1

- L4T R32.2


## Guide

```
vim ~/.bashrc

export OPENBLAS_CORETYPE=ARMV8

#converting darkenet weights to onnx weights
python3 yolov3_to_onnx.py --cfg ${CFG_PATH} --onnx ${ONNX_PATH} --num_class ${num_of_classes}

#building trt model and run inference
python3 onnx_to_tensorrt.py --cfg ${CFG_PATH} --onnx ${ONNX_PATH} --num_class ${num_of_classes} --input_img &{test_img_path}

```
