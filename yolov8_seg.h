#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>
#include "yolov8_utils.h"

class Yolov8Seg {
public:
	Yolov8Seg() {
	}
	~Yolov8Seg() {}

	bool ReadModel(cv::dnn::Net& net, std::string& netPath, bool isCuda);
	bool Detect(cv::Mat& srcImg, cv::dnn::Net& net, std::vector<OutputSeg>& output);

#if(defined YOLO_P6 && YOLO_P6==true)
	//const float _netAnchors[4][6] = { { 19,27, 44,40, 38,94 },{ 96,68, 86,152, 180,137 },{ 140,301, 303,264, 238,542 },{ 436,615, 739,380, 925,792 } };

	const int _netWidth = 1280;  //ONNX图片输入宽度
	const int _netHeight = 1280; //ONNX图片输入高度
	int _segWidth = 320;  //_segWidth=_netWidth/mask_ratio
	int _segHeight = 320;
	int _segChannels = 32;

#else
	//const float _netAnchors[3][6] = { { 10,13, 16,30, 33,23 },{ 30,61, 62,45, 59,119 },{ 116,90, 156,198, 373,326 } };
	
	const int _netWidth = 640;   //ONNX图片输入宽度
	const int _netHeight = 640;  //ONNX图片输入高度
	 int _segWidth = 160;    //_segWidth=_netWidth/mask_ratio
	 int _segHeight = 160;
	 int _segChannels = 32;

#endif // YOLO_P6
	
	float _classThreshold = 0.25;
	float _nmsThreshold = 0.45;
	float _maskThreshold = 0.5;


	//类别名，自己的模型需要修改此项
	std::vector<std::string> _className = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush" };
};
