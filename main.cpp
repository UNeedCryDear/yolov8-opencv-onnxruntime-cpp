#include <iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
#include "yolov8.h"
#include "yolov8_onnx.h"
#include "yolov8_seg.h"
#include "yolov8_seg_onnx.h"
#include<time.h>

using namespace std;
using namespace cv;
using namespace dnn;

template<typename _Tp>
int yolov8(_Tp& cls)
{
	string img_path = "./images/bus.jpg";
	string model_path = "./models/yolov8s-seg.onnx";
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
	Mat img = imread(img_path);
	clock_t t1, t2;
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
int yolov8_onnx(_Tp& cls)
{
	string img_path = "./images/bus.jpg";
	string model_path = "./models/yolov8s-seg.onnx";


	if (cls.ReadModel( model_path, true)) {
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
	Mat img = imread(img_path);
	clock_t t1, t2;
	if (cls.OnnxDetect(img, result)) {
		DrawPred(img, result, cls._className, color);
	}
	else {
		cout << "Detect Failed!" << endl;
	}
	system("pause");
	return 0;
}


int main() {
	Yolov8 task_detect;
	Yolov8Seg task_segment;

	Yolov8Onnx task_detect_onnx;
	Yolov8SegOnnx task_segment_onnx;

	yolov8(task_detect);    //Opencv detect
	yolov8(task_segment);   //opencv segment
	yolov8_onnx(task_detect_onnx);  //onnxruntime detect
	yolov8_onnx(task_segment_onnx); //onnxruntime segment

	return 0;
}


