# yolov8-opencv-onnxruntime-cpp
## 使用OpenCV-dnn和ONNXRuntime部署yolov8目标检测和实例分割模型<br>
基于yolov8:https://github.com/ultralytics/ultralytics

## requirements for opencv-dnn
1. > OpenCV>=4.7.0<br>
OpenCV>=4.7.0<br>
OpenCV>=4.7.0<br>

2. export for opencv-dnn:</br>
```bash
#Note: When exporting to opencv, it is best to set opset to 12

yolo export model=path/to/model.pt format=onnx dynamic=False  opset=12
```

3. export RT-DETR:</br>
```bash
#Note: rtdetr need opset>=16,dynamic=False/True 

yolo export model=path/to/rtdetr-l.pt format=onnx  opset=16

```

```python
from ultralytics import YOLO
model = YOLO('./pre_model/yolov8-rtdetr-l.pt')
results = model.export(format='onnx',opset=16)
```


## requirements for onnxruntime （only yolo*_onnx.h/cpp）
>opencv>=4.5.0 </br>
ONNXRuntime>=1.9.0 </br>

## 更新说明：
#### 2024.04.15更新<br>
+ 新增yolov8-pose模型部署（https://github.com/UNeedCryDear/yolov8-opencv-onnxruntime-cpp/issues/52）
+ 修复命名空间使用问题。

#### 2024.01.22更新<br>
+ 新增yolov8-obb模型部署（https://github.com/UNeedCryDear/yolov8-opencv-onnxruntime-cpp/issues/40）
+ 修改一些便利性使用的问题。
#### 2023.12.05更新<br>
+ 新增yolov8-RTDETR部署。
+ 优化部分代码，例如输出shape之类从输出中获取，而非像之前需要设置正确参数。

#### 2023.11.09更新<br>
+ 修复此pr中提到的一些问题[https://github.com/UNeedCryDear/yolov8-opencv-onnxruntime-cpp/pull/30]，此bug会导致mask与box大小可能会差几个像素从而导致出现一些问题（如果用的时候没有注意的话），本次更新之后会将其缩放到一致大小。
+ 新增视频流推理的demo，这是由于发现很多初学者调用视频的时候总是每一张图片都去读取一次模型，所以本次更新一起加上去。

#### 2023.09.20更新<br>
+ 0.新增模型路径检查，部分issue查了半天，发现模型路径不对。
+ 1.计算mask部分bug修复，此前如果输入大小非640的话，需要同时设置头文件和结构体才能完成检测，但是大部分人只修改了一个地方，目前优化这部分内容，只需要修头文件中的定义即可。另外将segHeight和segWidth设置为从网络输出中读取，这样如果mask-ratio不是4倍的话，可以不需要修改这两个参数值。
+ 2.修复```GetMask2()```中可能导致越界的问题。<br>


#### 2023.02.17更新<br>
+ 0.新增加onnxruntime旧版本API接口支持
+ 1.opencv不支持动态推理，请将dymanic设置为False导出onnx,同时opset需要设置为12。
+ 2.关于换行符，windows下面需要设置为CRLF，上传到github会自动切换成LF，windows下面切换一下即可<br>

#### 2023.02.07 更新：</br>
+ yolov8使用opencv-dnn推理的话，目前只支持opencv4.7.0及其以上的版本，我暂时也没找到怎么修改适应opencv4.5.0的版本（￣へ￣），这个版本需求和onnxruntime无关，onnxruntime只需要4.5.0的版本,4.x的版本应该都可以用，只要能正确读取，有```cv::dnn::blobFromImages()```这个函数即可,如果真的没有这个函数，你自己将其源码抠出来用也是可以的，或者大佬们自己实现该函数功能。
+ 而目前opencv4.7.0的版本有问题（https://github.com/opencv/opencv/issues/23080) ，如果你的CPU不支持```AVX2```指令集，则需要在```net.forward()``` 前面加上```net.enableWinograd(false);```来关闭Winograd加速，如果支持这个指令集的话可以开启加速（蚊子腿）。

依照惯例贴一张yolov8-seg.onnx在640x640下用onnxruntime运行结果图：
![Alt text](images/bus_out.bmp)
