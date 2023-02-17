#pragma once
#include<iostream>
#include <numeric>
#include<opencv2/opencv.hpp>

#define YOLO_P6 false //是否使用P6模型
#define ORT_OLD_VISON 12  //ort1.12.0 之前的版本为旧版本API
struct OutputSeg {
	int id;             //结果类别id
	float confidence;   //结果置信度
	cv::Rect box;       //矩形框
	cv::Mat boxMask;       //矩形框内mask，节省内存空间和加快速度
};
struct MaskParams {
	int segChannels = 32;
	int segWidth = 160;
	int segHeight = 160;
	int netWidth = 640;
	int netHeight = 640;
	float maskThreshold = 0.5;
	cv::Size srcImgShape;
	cv::Vec4d params;

};
bool CheckParams(int netHeight, int netWidth, const int* netStride, int strideSize); 
void DrawPred(cv::Mat& img, std::vector<OutputSeg> result, std::vector<std::string> classNames, std::vector<cv::Scalar> color);
void LetterBox(const cv::Mat& image, cv::Mat& outImage,
	cv::Vec4d& params, //[ratio_x,ratio_y,dw,dh]
	const cv::Size& newShape = cv::Size(640, 640),
	bool autoShape = false,
	bool scaleFill = false,
	bool scaleUp = true,
	int stride = 32,
	const cv::Scalar& color = cv::Scalar(114, 114, 114));
void GetMask(const cv::Mat& maskProposals, const cv::Mat& maskProtos, std::vector<OutputSeg>& output, const MaskParams& maskParams);
void GetMask2(const cv::Mat& maskProposals, const cv::Mat& maskProtos, OutputSeg& output, const MaskParams& maskParams);



