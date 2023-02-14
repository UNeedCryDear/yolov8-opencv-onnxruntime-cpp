# yolov8-opencv-onnxruntime-cpp
## 使用OpenCV-dnn和ONNXRuntime部署yolov8目标检测和实例分割模型<br>
基于yolov8:https://github.com/ultralytics/ultralytics

**！！！！！！！！！！！！！！！**<br>
**OpenCV>=4.7.0**<br>
**OpenCV>=4.7.0**<br>
**OpenCV>=4.7.0**<br>
for opencv-dnn:</br>
```yolo export model=yolov8n.pt format=onnx dynamic=False  opset=12```</br>
**！！！！！！！！！！！！！！！**<br>
**ONNXRuntime>=?**</br>

#### 2023.02.07 更新：</br>
+ yolov8目前只支持opencv4.7.0以上的版本，我暂时也没找到怎么修改适应opencv4.5.0的版本（￣へ￣）
+ 而目前opencv4.7.0的版本有问题（https://github.com/opencv/opencv/issues/23080) ，如果你的CPU不支持```AVX2```指令集，则需要在```net.forward()``` 前面加上```net.enableWinograd(false);```来关闭Winograd加速，如果支持这个指令集的话可以开启加速（蚊子腿）。

依照惯例贴一张yolov8-seg.onnx在640x640下用onnxruntime运行结果图：
![Alt text](images/bus_out.bmp)
