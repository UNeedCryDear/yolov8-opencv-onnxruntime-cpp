#include"yolov8_pose.h"

//using namespace std;
//using namespace cv;
//using namespace cv::dnn;

bool Yolov8Pose::ReadModel(cv::dnn::Net& net, std::string& netPath, bool isCuda = false) {
	try {
		if (!CheckModelPath(netPath))
			return false;
#if CV_VERSION_MAJOR==4 &&CV_VERSION_MINOR<7
		std::cout << "OBB Need OpenCV Version >=4.7.0" << std::endl; 
		return false;
#endif
		net = cv::dnn::readNetFromONNX(netPath);
#if CV_VERSION_MAJOR==4 &&CV_VERSION_MINOR==7&&CV_VERSION_REVISION==0
		net.enableWinograd(false);  //bug of opencv4.7.x in AVX only platform ,https://github.com/opencv/opencv/pull/23112 and https://github.com/opencv/opencv/issues/23080 
		//net.enableWinograd(true);		//If your CPU supports AVX2, you can set it true to speed up
#endif
	}
	catch (const std::exception&) {
		return false;
	}

	if (isCuda) {
		//cuda
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA); //or DNN_TARGET_CUDA_FP16
	}
	else {
		//cpu
		std::cout << "Inference device: CPU" << std::endl;
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	return true;
}


bool Yolov8Pose::Detect(cv::Mat& srcImg, cv::dnn::Net& net, std::vector<OutputParams>& output) {
	cv::Mat blob;
	output.clear();
	int col = srcImg.cols;
	int row = srcImg.rows;
	cv::Mat netInputImg;
	cv::Vec4d params;
	LetterBox(srcImg, netInputImg, params, cv::Size(_netWidth, _netHeight));
	cv::dnn::blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(0, 0, 0), true, false);
	//**************************************************************************************************************************************************/
	//如果在其他设置没有问题的情况下但是结果偏差很大，可以尝试下用下面两句语句
	// If there is no problem with other settings, but results are a lot different from  Python-onnx , you can try to use the following two sentences
	// 
	//$ cv::dnn::blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(104, 117, 123), true, false);
	//$ cv::dnn::blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(114, 114,114), true, false);
	//****************************************************************************************************************************************************/
	net.setInput(blob);
	std::vector<cv::Mat> net_output_img;
	net.forward(net_output_img, net.getUnconnectedOutLayersNames()); //get outputs
	std::vector<int> class_ids;// res-class_id
	std::vector<float> confidences;// res-conf 
	std::vector<cv::Rect> boxes;// res-box
	std::vector<std::vector<PoseKeyPoint>> pose_key_points;
	cv::Mat output0 = cv::Mat(cv::Size(net_output_img[0].size[2], net_output_img[0].size[1]), CV_32F, (float*)net_output_img[0].data).t();  //[bs,20,21504]=>[bs,21504,20]
	int net_width = output0.cols;
	int rows = output0.rows;
	int key_point_length = net_width - 5;
	int key_point_num = 17; //_bodyKeyPoints.size(), shape[x, y, confidence]
	if (key_point_num * 3 != key_point_length) {
		std::cout << "Pose should be shape [x, y, confidence] with 17-points" << std::endl;
		return false;
	}

	float* pdata = (float*)output0.data;
	for (int r = 0; r < rows; ++r) {
		float max_class_socre=pdata[4];
		if (max_class_socre >= _classThreshold) {
			//rect [x,y,w,h]
			float x = (pdata[0] - params[2]) / params[0];
			float y = (pdata[1] - params[3]) / params[1];
			float w = pdata[2] / params[0];
			float h = pdata[3] / params[1];
			class_ids.push_back(0);
			confidences.push_back(max_class_socre);
			int left = MAX(int(x - 0.5 * w + 0.5), 0);
			int top = MAX(int(y - 0.5 * h + 0.5), 0);
			boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
			std::vector<PoseKeyPoint> temp_kpts;
			for (int kpt = 0; kpt < key_point_length; kpt += 3) {
				PoseKeyPoint temp_kp;
				temp_kp.x = (pdata[5 + kpt] - params[2]) / params[0];
				temp_kp.y = (pdata[6 + kpt] - params[3]) / params[1];
				temp_kp.confidence = pdata[7 + kpt];
				temp_kpts.push_back(temp_kp);
			}
			pose_key_points.push_back(temp_kpts);

		}
		pdata += net_width;//next line
	}
	//NMS
	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);
	std::vector<std::vector<float>> temp_mask_proposals;
	cv::Rect holeImgRect(0, 0, srcImg.cols, srcImg.rows);
	for (int i = 0; i < nms_result.size(); ++i) {
		int idx = nms_result[i];
		OutputParams result;
		result.id = class_ids[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx]& holeImgRect;
		result.keyPoints = pose_key_points[idx];
		output.push_back(result);

	}
	if (output.size())
		return true;
	else
		return false;
}

