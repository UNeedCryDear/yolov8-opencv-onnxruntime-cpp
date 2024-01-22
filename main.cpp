#include <iostream>
#include<opencv2/opencv.hpp>

#include<math.h>
#include "yolov8.h"
#include "yolov8_onnx.h"
#include "yolov8_seg.h"
#include "rtdetr_onnx.h"
#include "yolov8_seg_onnx.h"
#include "yolov8_obb.h"
#include "yolov8_obb_onnx.h"
#include<time.h>
//#define  VIDEO_OPENCV //if define, use opencv for video.

using namespace std;
using namespace cv;
using namespace dnn;

template<typename _Tp>
int yolov8(_Tp& task, Mat& img, string& model_path)
{

	Net net;
	if (task.ReadModel(net, model_path, false)) {
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
	vector<OutputParams> result;


	if (task.Detect(img, net, result)) {
		DrawPred(img, result, task._className, color);
	}
	else {
		cout << "Detect Failed!" << endl;
	}
	system("pause");
	return 0;
}

template<typename _Tp>
int yolov8_onnx(_Tp& task, Mat& img, string& model_path)
{

	if (task.ReadModel(model_path, false)) {
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
	vector<OutputParams> result;
	if (task.OnnxDetect(img, result)) {
		DrawPred(img, result, task._className, color);
	}
	else {
		cout << "Detect Failed!" << endl;
	}
	system("pause");
	return 0;
}


template<typename _Tp>
int video_demo(_Tp& task, string& model_path)
{
	vector<Scalar> color;
	srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
	}
	vector<OutputParams> result;
	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		std::cout << "open capture failured!" << std::endl;
		return -1;
	}
	Mat frame;
#ifdef VIDEO_OPENCV
	Net net;
	if (task.ReadModel(net, model_path, true)) {
		cout << "read net ok!" << endl;
	}
	else {
		cout << "read net failured!" << endl;
		return -1;
	}

#else
	if (task.ReadModel(model_path, true)) {
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

		if (task.Detect(frame, net, result)) {
			DrawPred(frame, result, task._className, color, true);
		}
#else
		if (task.OnnxDetect(frame, result)) {
			DrawPred(frame, result, task._className, color, true);
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

	string img_path = "./images/bus.jpg";

	string model_path_detect = "./models/yolov8s.onnx";
	string model_path_rtdetr = "./models/rtdetr-l.onnx";  //yolov8-redetr
	string model_path_obb = "./models/yolov8s-obb.onnx";
	string model_path_seg = "./models/yolov8s-seg.onnx";


	Mat src = imread(img_path);
	Mat img = src.clone();

	Yolov8				task_detect_ocv;
	Yolov8Onnx			task_detect_ort;
	RTDETROnnx			task_rtdetr_ort;
	Yolov8Seg			task_segment_ocv;
	Yolov8SegOnnx		task_segment_ort;
	Yolov8Obb			task_obb_ocv;
	Yolov8ObbOnnx		task_obb_ort;

	yolov8(task_detect_ocv,img,model_path_detect);    //yolov8 opencv detect
	//img = src.clone();
	//yolov8_onnx(task_detect_ort,img,model_path_detect);  //yoolov8 onnxruntime detect
	// 
	//img = src.clone();
	//yolov8_onnx(task_rtdetr_ort, img, model_path_rtdetr);  //yolov8-rtdetr onnxruntime detect

	//img = src.clone();
	//yolov8(task_segment_ocv,img,model_path_seg);   //yolov8 opencv segment
	//img = src.clone();
	//yolov8_onnx(task_segment_ort,img,model_path_seg); //yolov8 onnxruntime segment


	//img = src.clone();
	//yolov8(task_obb_ocv, img, model_path_obb); //yolov8 opencv obb
	//img = src.clone();
	//yolov8_onnx(task_obb_ort, img, model_path_obb); //yolov8 onnxruntime obb

#ifdef VIDEO_OPENCV
	video_demo(task_detect_ocv, model_path_detect);
	//video_demo(task_segment_ocv, model_path_seg);
#else
	//video_demo(task_detect_ort, model_path_detect);
	//video_demo(task_rtdetr_ort, model_path_rtdetr);
	//video_demo(task_segment_ort, model_path_seg);
#endif
	return 0;
}


