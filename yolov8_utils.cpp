#pragma once
#include "yolov8_utils.h"
//using namespace cv;
//using namespace std;
bool CheckParams(int netHeight, int netWidth, const int* netStride, int strideSize) {
	if (netHeight % netStride[strideSize - 1] != 0 || netWidth % netStride[strideSize - 1] != 0)
	{
		std::cout << "Error:_netHeight and _netWidth must be multiple of max stride " << netStride[strideSize - 1] << "!" << std::endl;
		return false;
	}
	return true;
}
bool CheckModelPath(std::string modelPath) {
	if (0 != _access(modelPath.c_str(), 0)) {
		std::cout << "Model path does not exist,  please check " << modelPath << std::endl;
		return false;
	}
	else
		return true;

}
void LetterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape,
	bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar& color)
{
	if (false) {
		int maxLen = MAX(image.rows, image.cols);
		outImage = cv::Mat::zeros(cv::Size(maxLen, maxLen), CV_8UC3);
		image.copyTo(outImage(cv::Rect(0, 0, image.cols, image.rows)));
		params[0] = 1;
		params[1] = 1;
		params[3] = 0;
		params[2] = 0;
	}

	cv::Size shape = image.size();
	float r = std::min((float)newShape.height / (float)shape.height,
		(float)newShape.width / (float)shape.width);
	if (!scaleUp)
		r = std::min(r, 1.0f);

	float ratio[2]{ r, r };
	int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

	auto dw = (float)(newShape.width - new_un_pad[0]);
	auto dh = (float)(newShape.height - new_un_pad[1]);

	if (autoShape)
	{
		dw = (float)((int)dw % stride);
		dh = (float)((int)dh % stride);
	}
	else if (scaleFill)
	{
		dw = 0.0f;
		dh = 0.0f;
		new_un_pad[0] = newShape.width;
		new_un_pad[1] = newShape.height;
		ratio[0] = (float)newShape.width / (float)shape.width;
		ratio[1] = (float)newShape.height / (float)shape.height;
	}

	dw /= 2.0f;
	dh /= 2.0f;

	if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
	{
		cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
	}
	else {
		outImage = image.clone();
	}

	int top = int(std::round(dh - 0.1f));
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(dw - 0.1f));
	int right = int(std::round(dw + 0.1f));
	params[0] = ratio[0];
	params[1] = ratio[1];
	params[2] = left;
	params[3] = top;
	cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void GetMask(const cv::Mat& maskProposals, const cv::Mat& maskProtos, std::vector<OutputParams>& output, const MaskParams& maskParams) {
	//std::cout << maskProtos.size << std::endl;

	int net_width = maskParams.netWidth;
	int net_height = maskParams.netHeight;
	int seg_channels = maskProtos.size[1];
	int seg_height = maskProtos.size[2];
	int seg_width = maskProtos.size[3];
	float mask_threshold = maskParams.maskThreshold;
	cv::Vec4f params = maskParams.params;
	cv::Size src_img_shape = maskParams.srcImgShape;

	cv::Mat protos = maskProtos.reshape(0, { seg_channels,seg_width * seg_height });

	cv::Mat matmul_res = (maskProposals * protos).t();
	cv::Mat masks = matmul_res.reshape(output.size(), { seg_width,seg_height });
	std::vector<cv::Mat> maskChannels;
	split(masks, maskChannels);
	for (int i = 0; i < output.size(); ++i) {
		cv::Mat dest, mask;
		//sigmoid
		cv::exp(-maskChannels[i], dest);
		dest = 1.0 / (1.0 + dest);

		cv::Rect roi(int(params[2] / net_width * seg_width), int(params[3] / net_height * seg_height), int(seg_width - params[2] / 2), int(seg_height - params[3] / 2));
		dest = dest(roi);
		resize(dest, mask, src_img_shape, cv::INTER_NEAREST);

		//crop
		cv::Rect temp_rect = output[i].box;
		mask = mask(temp_rect) > mask_threshold;
		output[i].boxMask = mask;
	}
}

void GetMask2(const cv::Mat& maskProposals, const cv::Mat& maskProtos, OutputParams& output, const MaskParams& maskParams) {
	int net_width = maskParams.netWidth;
	int net_height = maskParams.netHeight;
	int seg_channels = maskProtos.size[1];
	int seg_height = maskProtos.size[2];
	int seg_width = maskProtos.size[3];
	float mask_threshold = maskParams.maskThreshold;
	cv::Vec4f params = maskParams.params;
	cv::Size src_img_shape = maskParams.srcImgShape;

	cv::Rect temp_rect = output.box;
	//crop from mask_protos
	int rang_x = floor((temp_rect.x * params[0] + params[2]) / net_width * seg_width);
	int rang_y = floor((temp_rect.y * params[1] + params[3]) / net_height * seg_height);
	int rang_w = ceil(((temp_rect.x + temp_rect.width) * params[0] + params[2]) / net_width * seg_width) - rang_x;
	int rang_h = ceil(((temp_rect.y + temp_rect.height) * params[1] + params[3]) / net_height * seg_height) - rang_y;

	//如果下面的 mask_protos(roi_rangs).clone()位置报错，说明你的output.box数据不对，或者矩形框就1个像素的，开启下面的注释部分防止报错。
	rang_w = MAX(rang_w, 1);
	rang_h = MAX(rang_h, 1);
	if (rang_x + rang_w > seg_width) {
		if (seg_width - rang_x > 0)
			rang_w = seg_width - rang_x;
		else
			rang_x -= 1;
	}
	if (rang_y + rang_h > seg_height) {
		if (seg_height - rang_y > 0)
			rang_h = seg_height - rang_y;
		else
			rang_y -= 1;
	}

	std::vector<cv::Range> roi_rangs;
	roi_rangs.push_back(cv::Range(0, 1));
	roi_rangs.push_back(cv::Range::all());
	roi_rangs.push_back(cv::Range(rang_y, rang_h + rang_y));
	roi_rangs.push_back(cv::Range(rang_x, rang_w + rang_x));

	//crop
	cv::Mat temp_mask_protos = maskProtos(roi_rangs).clone();
	cv::Mat protos = temp_mask_protos.reshape(0, { seg_channels,rang_w * rang_h });
	cv::Mat matmul_res = (maskProposals * protos).t();
	cv::Mat masks_feature = matmul_res.reshape(1, { rang_h,rang_w });
	cv::Mat dest, mask;

	//sigmoid
	cv::exp(-masks_feature, dest);
	dest = 1.0 / (1.0 + dest);

	int left = floor((net_width / seg_width * rang_x - params[2]) / params[0]);
	int top = floor((net_height / seg_height * rang_y - params[3]) / params[1]);
	int width = ceil(net_width / seg_width * rang_w / params[0]);
	int height = ceil(net_height / seg_height * rang_h / params[1]);

	resize(dest, mask, cv::Size(width, height), cv::INTER_NEAREST);
	cv::Rect mask_rect = temp_rect - cv::Point(left, top);
	mask_rect &= cv::Rect(0, 0, width, height);
	mask = mask(mask_rect) > mask_threshold;
	if (mask.rows != temp_rect.height || mask.cols != temp_rect.width) { //https://github.com/UNeedCryDear/yolov8-opencv-onnxruntime-cpp/pull/30
		resize(mask, mask, temp_rect.size(), cv::INTER_NEAREST);
	}
	output.boxMask = mask;

}
int BBox2Obb(float centerX, float centerY, float boxW, float boxH, float angle, cv::RotatedRect& rotatedRect) {
	rotatedRect = cv::RotatedRect(cv::Point2f(centerX, centerY), cv::Size2f(boxW, boxH),angle);
	return 0;
}
void DrawRotatedBox(cv::Mat &srcImg, cv::RotatedRect box, cv::Scalar color, int thinkness) {
	cv::Point2f p[4];
	box.points(p);
	for (int l = 0; l < 4; ++l) {
		line(srcImg, p[l], p[(l + 1) % 4], color, thinkness, 8);
	}
}

void DrawPred(cv::Mat& img, std::vector<OutputParams> result, std::vector<std::string> classNames, std::vector<cv::Scalar> color, bool isVideo) {
	cv::Mat mask = img.clone();
	for (int i = 0; i < result.size(); i++) {
		int left=0, top=0;

		int color_num = i;
		if (result[i].box.area() > 0) {
			rectangle(img, result[i].box, color[result[i].id], 2, 8);
			left = result[i].box.x;
			top = result[i].box.y;
		}
		if (result[i].rotatedBox.size.width * result[i].rotatedBox.size.height > 0) {
			DrawRotatedBox(img, result[i].rotatedBox, color[result[i].id], 2);
			left = result[i].rotatedBox.center.x;
			top = result[i].rotatedBox.center.y;
		}
		if (result[i].boxMask.rows && result[i].boxMask.cols > 0)
			mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);
		std::string label = classNames[result[i].id] + ":" + std::to_string(result[i].confidence);
		int baseLine;
		cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = MAX(top, labelSize.height);
		//rectangle(frame, cv::Point(left, top - int(1.5 * labelSize.height)), cv::Point(left + int(1.5 * labelSize.width), top + baseLine), cv::Scalar(0, 255, 0), FILLED);
		putText(img, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
	}
	cv::addWeighted(img, 0.5, mask, 0.5, 0, img); //add mask to src
	cv::imshow("1", img);
	if (!isVideo)
		cv::waitKey();
	//destroyAllWindows();

}


void DrawPredPose(cv::Mat& img, std::vector<OutputParams> result, PoseParams& poseParams, bool isVideo) {
	for (int i = 0; i < result.size(); i++) {
		int left, top;
		int color_num = i;
		if (result[i].box.area() > 0) {
			rectangle(img, result[i].box, poseParams.personColor, 2, 8);
			left = result[i].box.x;
			top = result[i].box.y;
		}
		else 
			continue;
		
		std::string label =  "person:" + std::to_string(result[i].confidence);
		int baseLine;
		cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = MAX(top, labelSize.height);
		//rectangle(frame, cv::Point(left, top - int(1.5 * labelSize.height)), cv::Point(left + int(1.5 * labelSize.width), top + baseLine), cv::Scalar(0, 255, 0), FILLED);
		putText(img, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 1, poseParams.personColor, 2);
		if (result[i].keyPoints.size()!= poseParams.kptBodyNames.size())
			continue;
		for (int j = 0; j < result[i].keyPoints.size(); ++j) {
			PoseKeyPoint kpt = result[i].keyPoints[j];
			if (kpt.confidence < poseParams.kptThreshold)
				continue;
			cv::Scalar kptColor = poseParams.posePalette[poseParams.kptColor[j]];
			cv::circle(img, cv::Point(kpt.x, kpt.y), poseParams.kptRadius, kptColor, -1, 8);

		}
		if (poseParams.isDrawKptLine) {
			for (int j = 0; j < poseParams.skeleton.size(); ++j) {
				PoseKeyPoint kpt0 = result[i].keyPoints[poseParams.skeleton[j][0]-1];
				PoseKeyPoint kpt1 = result[i].keyPoints[poseParams.skeleton[j][1]-1];
				if (kpt0.confidence < poseParams.kptThreshold || kpt1.confidence < poseParams.kptThreshold)
					continue;
				cv::Scalar kptColor= poseParams.posePalette[poseParams.limbColor[j]];
				cv::line(img, cv::Point(kpt0.x, kpt0.y),cv::Point(kpt1.x, kpt1.y),  kptColor, 2, 8);
			}
		}
	}
	cv::imshow("1", img);
	if (!isVideo)
		cv::waitKey();
	//destroyAllWindows();

}
