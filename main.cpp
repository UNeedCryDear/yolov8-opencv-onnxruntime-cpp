#include <iostream>
#include<opencv2/opencv.hpp>

#include<math.h>
#include "yolov8.h"
#include "yolov8_onnx.h"
#include "yolov8_seg.h"
#include "rtdetr_onnx.h"
#include "yolov8_seg_onnx.h"
#include<time.h>
#define  VIDEO_OPENCV //if define, use opencv for video.

using namespace std;
using namespace cv;
using namespace dnn;

template<typename _Tp>
int yolov8(_Tp& cls,Mat& img,string& model_path)
{

	Net net;
	if (cls.ReadModel(net, model_path, false)) {
		cout << "read net ok!" << endl;
	}
	else {
		return -1;
	}
	//生成随机颜色
	vector<Scalar> color;
	srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
	}
	vector<OutputSeg> result;


	if (cls.Detect(img, net, result)) {
		DrawPred(img, result, cls._className, color);
	}
	else {
		cout << "Detect Failed!" << endl;
	}
	system("pause");
	return 0;
}

template<typename _Tp>
int yolov8_onnx(_Tp& cls, Mat& img, string& model_path)
{

	if (cls.ReadModel( model_path, false)) {
		cout << "read net ok!" << endl;
	}
	else {
		return -1;
	}
	//生成随机颜色
	vector<Scalar> color;
	srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
	}
	vector<OutputSeg> result;
	if (cls.OnnxDetect(img, result)) {
		DrawPred(img, result, cls._className, color);
	}
	else {
		cout << "Detect Failed!" << endl;
	}
	system("pause");
	return 0;
}


template<typename _Tp>
int video_demo(_Tp& cls, string& model_path)
{
	vector<Scalar> color;
	srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
	}
	vector<OutputSeg> result;
	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		std::cout << "open capture failured!" << std::endl;
		return -1;
	}
	Mat frame;
#ifdef VIDEO_OPENCV
	Net net;
	if (cls.ReadModel(net, model_path, true)) {
		cout << "read net ok!" << endl;
	}
	else {
		cout << "read net failured!" << endl;
		return -1;
	}

#else
	if (cls.ReadModel(model_path, true)) {
		cout << "read net ok!" << endl;
	}
	else {
		cout << "read net failured!" << endl;
		return -1;
	}

#endif

	while (true)
	{

		cap.read(frame);
		if (frame.empty())
		{
			std::cout << "read to end" << std::endl;
			break;
		}
		result.clear();
#ifdef VIDEO_OPENCV

		if (cls.Detect(frame, net, result)) {
			DrawPred(frame, result, cls._className, color, true);
		}
#else
		if (cls.OnnxDetect(frame, result)) {
			DrawPred(frame, result, cls._className, color, true);
		}
#endif
		int k = waitKey(10);
		if (k == 27) { //esc 
			break;
		}

	}
	cap.release();

	system("pause");

	return 0;
}


int main() {

	string img_path = "./images/zidane.jpg";
	string seg_model_path = "./models/yolov8s-seg.onnx";
	string detect_model_path = "./models/yolov8s.onnx";
	string detect_rtdetr_path = "./models/rtdetr-l.onnx";  //yolov8-redetr
	Mat src = imread(img_path);
	Mat img = src.clone();

	Yolov8 task_detect;
	Yolov8Onnx task_detect_onnx;
	RTDETROnnx task_detect_rtdetr_onnx;

	Yolov8Seg task_segment;
	Yolov8SegOnnx task_segment_onnx;

	yolov8(task_detect,img,detect_model_path);    //yolov8 opencv detect
	//img = src.clone();
	//yolov8_onnx(task_detect_onnx,img,detect_model_path);  //yoolov8 onnxruntime detect
	// 
	//img = src.clone();
	//yolov8_onnx(task_detect_rtdetr_onnx, img, detect_rtdetr_path);  //yolov8-rtdetr onnxruntime detect

	//img = src.clone();
	//yolov8(task_segment,img,seg_model_path);   //yolov8 opencv segment

	//img = src.clone();
	//yolov8_onnx(task_segment_onnx,img,seg_model_path); //yolov8 onnxruntime segment

#ifdef VIDEO_OPENCV
	video_demo(task_detect, detect_model_path);
	//video_demo(task_segment, seg_model_path);
#else
	//video_demo(task_detect_onnx, detect_model_path);
	//video_demo(task_detect_rtdetr_onnx, detect_rtdetr_path);
	//video_demo(task_segment_onnx, seg_model_path);
#endif
	return 0;
}


