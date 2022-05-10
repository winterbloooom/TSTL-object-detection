#!/usr/bin/env python2
#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#
import sys, os, time
import numpy as np
import tensorrt as trt
import onnx
import common
import argparse
from PIL import Image,ImageDraw

from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def parse_args():
    parser = argparse.ArgumentParser(description="yolov3_to_onnx")
    parser.add_argument('--cfg', dest='cfg', help="the path of model cfg",
                        default=None, type=str)
    parser.add_argument('--onnx', dest='onnx', help="the path of onnx model",
                        default=None, type=str)
    parser.add_argument('--num_class', dest='num_class', help="the number of class",
                        default=None, type=int)
    parser.add_argument('--input_img', dest='input_img', help="the path of input_img",
                        default=None, type=str)
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    return args

#parse width, height, masks and anchors from cfg file
def parse_cfg_wh(cfg):
    masks = []
    with open(cfg, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'width' in line:
                w = int(line[line.find('=')+1:].replace('\n',''))
            elif 'height' in line:
                h = int(line[line.find('=')+1:].replace('\n',''))
            elif 'anchors' in line:
                anchor = line.split('=')[1].replace('\n','')
                anc = [int(a) for a in anchor.split(',')]
                anchors = [(anc[i*2], anc[i*2+1]) for i in range(len(anc) // 2)]
            elif 'mask' in line:
                mask = line.split('=')[1].replace('\n','')
                m = tuple(int(a) for a in mask.split(','))
                masks.append(m)
    return w, h, masks, anchors

def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    draw = ImageDraw.Draw(image_raw)
    if bboxes is None and confidences is None and categories is None:
        return image_raw
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color)

    return image_raw

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        print("build engine")
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30 # 1GB
            builder.max_batch_size = 1
            print("fp16 :", builder.platform_has_fast_fp16)
            print("int8 :", builder.platform_has_fast_int8)
            print(builder.fp16_mode)
            builder.fp16_mode = True
            print(builder.fp16_mode)
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def main():
    """Create a TensorRT engine for ONNX-based YOLOv3(or YOLOv3-tiny) and run inference."""
    cfg_file_path = args.cfg
    num_class = args.num_class
    width, height, masks, anchors = parse_cfg_wh(cfg_file_path)
    # Try to load a previously generated YOLOv3 network graph in ONNX format:
    onnx_file_path = args.onnx
    engine_file_path = onnx_file_path.replace(".onnx", '.trt')
    print(onnx.checker.check_model(onnx_file_path))

    input_image_path = args.input_img

    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    input_resolution_yolov3_WH = (width, height)
    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(input_resolution_yolov3_WH)
    # Load an image from the specified input path, and return it together with  a pre-processed version
    image = preprocessor.process(input_image_path)
    # Store the shape of the original input image in WH format, we will need it for later
    shape_orig_WH = (image.shape[3], image.shape[2])

    # Output shapes expected by the post-processor
    output_channels = (num_class + 5) * 3
    if len(masks) == 2:
        output_shapes = [(1, output_channels, height//32, width//32), (1, output_channels, height//16, width//16)]
    else:
        output_shapes = [(1, output_channels, height//32, width//32), (1, output_channels, height//16, width//16), (1, output_channels, height//8, width//8)]

    postprocessor_args = {"yolo_masks": masks,                    # A list of 3 three-dimensional tuples for the YOLO masks
                          "yolo_anchors": anchors,
                          "obj_threshold": 0.5,                                               # Threshold for object coverage, float value between 0 and 1
                          "nms_threshold": 0.3,                                               # Threshold for non-max suppression algorithm, float value between 0 and 1
                          "yolo_input_resolution": input_resolution_yolov3_WH,
                          "num_class": num_class}

    postprocessor = PostprocessYOLO(**postprocessor_args)

    # Do inference with TensorRT
    trt_outputs = []
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        print('Running inference on image {}...'.format(input_image_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        start_time = time.time()
        inputs[0].host = image
        trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    print("infer : {}".format(time.time() - start_time))
    start_time = time.time()
    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

    # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
    boxes, classes, scores = postprocessor.process(trt_outputs, shape_orig_WH)
    print("latency : {}".format(time.time() - start_time))
    # Draw the bounding boxes onto the original input image and save it as a PNG file
    print(boxes, classes, scores)
    img_show = np.array(np.transpose(image[0], (1,2,0)) * 255, dtype=np.uint8)
    obj_detected_img = draw_bboxes(Image.fromarray(img_show), boxes, scores, classes, ALL_CATEGORIES)
    output_image_path = 'predict.png'
    obj_detected_img.save(output_image_path, 'PNG')
    print('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))

if __name__ == '__main__':
    args = parse_args()
    main()
