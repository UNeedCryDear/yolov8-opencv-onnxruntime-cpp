#include <iostream>
#include <opencv2/opencv.hpp>

#include "rtdetr_onnx.h"
#include "yolov8.h"
#include "yolov8_obb.h"
#include "yolov8_obb_onnx.h"
// #include "yolov8_onnx.h"
#include "yolov8_pose.h"
#include "yolov8_pose_onnx.h"
#include "yolov8_seg.h"
#include "yolov8_seg_onnx.h"
#include <math.h>
#include <time.h>
// #define  VIDEO_OPENCV //if define, use opencv for video.

using namespace std;
using namespace cv;
using namespace dnn;

template <typename _Tp>
int yolov8(_Tp &task, cv::Mat &img, std::string &model_path) {

  cv::dnn::Net net;
  if (task.ReadModel(net, model_path, false)) {
    std::cout << "read net ok!" << std::endl;
  } else {
    return -1;
  }
  // 生成随机颜色
  std::vector<cv::Scalar> color;
  srand(time(0));
  for (int i = 0; i < 80; i++) {
    int b = rand() % 256;
    int g = rand() % 256;
    int r = rand() % 256;
    color.push_back(cv::Scalar(b, g, r));
  }
  std::vector<OutputParams> result;

  bool isPose = false;
  if (typeid(task) == typeid(Yolov8Pose)) {
    isPose = true;
  }
  PoseParams poseParams;
  if (task.Detect(img, net, result)) {

    if (isPose)
      DrawPredPose(img, result, poseParams);
    else
      DrawPred(img, result, task._className, color);

  } else {
    std::cout << "Detect Failed!" << std::endl;
  }
  int _ = system("pause");
  return 0;
}

template <typename _Tp>
int yolov8_onnx(_Tp &task, cv::Mat &img, std::string &model_path) {

  if (task.ReadModel(model_path, false)) {
    std::cout << "read net ok!" << std::endl;
  } else {
    return -1;
  }
  // 生成随机颜色
  std::vector<cv::Scalar> color;
  srand(time(0));
  for (int i = 0; i < 80; i++) {
    int b = rand() % 256;
    int g = rand() % 256;
    int r = rand() % 256;
    color.push_back(cv::Scalar(b, g, r));
  }
  bool isPose = false;
  if (typeid(task) == typeid(Yolov8PoseOnnx)) {
    isPose = true;
  }
  PoseParams poseParams;

  std::vector<OutputParams> result;
  if (task.OnnxDetect(img, result)) {
    if (isPose)
      DrawPredPose(img, result, poseParams);
    else
      DrawPred(img, result, task._className, color);
  } else {
    std::cout << "Detect Failed!" << std::endl;
  }
  int _ = system("pause");
  return 0;
}

template <typename _Tp> int video_demo(_Tp &task, std::string &model_path) {
  std::vector<cv::Scalar> color;
  srand(time(0));
  for (int i = 0; i < 80; i++) {
    int b = rand() % 256;
    int g = rand() % 256;
    int r = rand() % 256;
    color.push_back(cv::Scalar(b, g, r));
  }
  std::vector<OutputParams> result;
  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
    std::cout << "open capture failured!" << std::endl;
    return -1;
  }
  cv::Mat frame;
  bool isPose = false;
  PoseParams poseParams;
#ifdef VIDEO_OPENCV
  cv::dnn::Net net;
  if (typeid(task) == typeid(Yolov8Pose)) {
    isPose = true;
  }
  if (task.ReadModel(net, model_path, true)) {
    std::cout << "read net ok!" << std::endl;
  } else {
    std::cout << "read net failured!" << std::endl;
    return -1;
  }

#else
  if (typeid(task) == typeid(Yolov8PoseOnnx)) {
    isPose = true;
  }
  if (task.ReadModel(model_path, true)) {
    std::cout << "read net ok!" << std::endl;
  } else {
    std::cout << "read net failured!" << std::endl;
    return -1;
  }

#endif

  while (true) {

    cap.read(frame);
    if (frame.empty()) {
      std::cout << "read to end" << std::endl;
      break;
    }
    result.clear();
#ifdef VIDEO_OPENCV

    if (task.Detect(frame, net, result)) {

      if (isPose)
        DrawPredPose(frame, result, poseParams, true);
      else
        DrawPred(frame, result, task._className, color, true);
    }
#else
    if (task.OnnxDetect(frame, result)) {
      if (isPose)
        DrawPredPose(frame, result, poseParams, true);
      else
        DrawPred(frame, result, task._className, color, true);
    }
#endif
    int k = waitKey(10);
    if (k == 27) { // esc
      break;
    }
  }
  cap.release();

  int _ = system("pause");

  return 0;
}

int main() {

  std::string img_path = "./images/bus.jpg";

  std::string model_path_detect = "./models/yolov8s-pose1.onnx";
  std::string model_path_rtdetr = "./models/rtdetr-l.onnx"; // yolov8-redetr
  std::string model_path_obb = "./models/yolov8s-obb.onnx";
  std::string model_path_seg = "./models/yolov8s-seg.onnx";
  std::string model_path_pose = "./models/yolov8s-pose.onnx";

  cv::Mat src = imread(img_path);
  cv::Mat img = src.clone();

  Yolov8 task_detect_ocv;
  Yolov8PoseOnnx task_detect_ort;

  Yolov8Seg task_segment_ocv;
  Yolov8SegOnnx task_segment_ort;

  Yolov8Obb task_obb_ocv;
  Yolov8ObbOnnx task_obb_ort;

  Yolov8Pose task_pose_ocv;
  Yolov8PoseOnnx task_pose_ort;

  RTDETROnnx task_rtdetr_ort;

  // yolov8(task_detect_ocv,img,model_path_detect);    //yolov8 opencv detect
  // img = src.clone();
  // yolov8_onnx(task_detect_ort,img,model_path_detect);  //yoolov8 onnxruntime
  // detect
  //
  // img = src.clone();
  // yolov8_onnx(task_rtdetr_ort, img, model_path_rtdetr);  //yolov8-rtdetr
  // onnxruntime detect

  // img = src.clone();
  // yolov8(task_segment_ocv,img,model_path_seg);   //yolov8 opencv segment
  // img = src.clone();
  // yolov8_onnx(task_segment_ort,img,model_path_seg); //yolov8 onnxruntime
  // segment

  // img = src.clone();
  // yolov8(task_obb_ocv, img, model_path_obb); //yolov8 opencv obb
  img = src.clone();
  yolov8_onnx(task_obb_ort, img, model_path_obb); // yolov8 onnxruntime obb

  //   img = src.clone();
  //   yolov8(task_pose_ocv, img, model_path_pose); // yolov8 opencv pose
  //   img = src.clone();
  //   yolov8_onnx(task_pose_ort, img, model_path_pose); // yolov8 onnxruntime
  //   pose

#ifdef VIDEO_OPENCV
  video_demo(task_detect_ocv, model_path_detect);
  // video_demo(task_segment_ocv, model_path_seg);
  // video_demo(task_pose_ocv, model_path_pose);
#else
  // video_demo(task_detect_ort, model_path_detect);
  // video_demo(task_rtdetr_ort, model_path_rtdetr);
  // video_demo(task_segment_ort, model_path_seg);
  // video_demo(task_pose_ort, model_path_pose);
#endif
  return 0;
}
