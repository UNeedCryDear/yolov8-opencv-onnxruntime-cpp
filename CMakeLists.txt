CMAKE_MINIMUM_REQUIRED(VERSION 3.26.0)
project(YOLOv8)

# use nix-build then comment the following lines
# SET (OpenCV_DIR /usr/include/opencv4)  #opencv root
# SET (ONNXRUNTIME_DIR path/to/onnxruntime)

FIND_PACKAGE(OpenCV REQUIRED)

ADD_EXECUTABLE(YOLOv8 yolov8.h yolov8_onnx.h yolov8_seg.h yolov8_seg_onnx.h yolov8_utils.h yolov8_obb_onnx.h
			 main.cpp yolov8.cpp yolov8_onnx.cpp yolov8_seg.cpp yolov8_seg_onnx.cpp yolov8_utils.cpp yolov8_obb_onnx.cpp)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)


TARGET_COMPILE_FEATURES(YOLOv8 PRIVATE cxx_std_14)
TARGET_LINK_LIBRARIES(YOLOv8 ${OpenCV_LIBS})

if (WIN32)
    TARGET_LINK_LIBRARIES(YOLOv8 "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
endif(WIN32)

if (UNIX)
    TARGET_LINK_LIBRARIES(YOLOv8 "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
    TARGET_LINK_LIBRARIES(YOLOv8 "${ONNXRUNTIME_DIR}/lib/libonnxruntime_providers_cuda.so")
    TARGET_LINK_LIBRARIES(YOLOv8 "${ONNXRUNTIME_DIR}/lib/libonnxruntime_providers_shared.so")


    install(FILES build/YOLOv8
        DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
    )

    # install(DIRECTORY psdk_lib/include
    # DESTINATION "${CMAKE_INSTALL_PREFIX}"
    # )
endif(UNIX)