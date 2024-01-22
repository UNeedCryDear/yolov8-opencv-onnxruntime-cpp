#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>
#include "yolov8_utils.h"

class Yolov8Obb {
public:
	Yolov8Obb() {
	}
	~Yolov8Obb() {}

	bool ReadModel(cv::dnn::Net& net, std::string& netPath, bool isCuda);
	bool Detect(cv::Mat& srcImg, cv::dnn::Net& net, std::vector<OutputParams>& output);

	int _netWidth = 1024;   //ONNX图片输入宽度
	int _netHeight = 1024;  //ONNX图片输入高度


	//类别名，自己的模型需要修改此项
	std::vector<std::string> _className = 
	{  "plane",  "ship", "storage tank", 
		"baseball diamond",  "tennis court",   "basketball court", 
		"ground track field", "harbor",  "bridge",  
		"large vehicle",  "small vehicle",  "helicopter",
		"roundabout",  "soccer ball field",  "swimming pool"
	};
private:
	float _classThreshold = 0.25;
	float _nmsThreshold = 0.45;
};
