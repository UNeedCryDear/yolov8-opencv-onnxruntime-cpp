CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)
project(YOLOv8)


SET (OpenCV_DIR path/to/opencv/build)  #opencv root
SET (ONNXRUNTIME_DIR path/to/onnxruntime)

FIND_PACKAGE(OpenCV REQUIRED)
#include_directories("")
ADD_EXECUTABLE(YOLOv8 yolov8.h yolov8_onnx.h yolov8_seg.h yolov8_seg_onnx.h yolov8_utils.h
			 main.cpp yolov8.cpp yolov8_onnx.cpp yolov8_seg.cpp yolov8_seg_onnx.cpp yolov8_utils.cpp)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

TARGET_INCLUDE_DIRECTORIES(YOLOv8 PRIVATE "${ONNXRUNTIME_DIR}/include")

TARGET_COMPILE_FEATURES(YOLOv8 PRIVATE cxx_std_14)
TARGET_LINK_LIBRARIES(YOLOv8 ${OpenCV_LIBS})

if (WIN32)
    TARGET_LINK_LIBRARIES(YOLOv8 "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
endif(WIN32)

if (UNIX)
    TARGET_LINK_LIBRARIES(YOLOv8 "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif(UNIX)