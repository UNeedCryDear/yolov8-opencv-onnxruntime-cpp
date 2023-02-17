#pragma once
#include <iostream>
#include<memory>
#include <opencv2/opencv.hpp>
#include "yolov8_utils.h"
#include<onnxruntime_cxx_api.h>

//#include <tensorrt_provider_factory.h>  //if use OrtTensorRTProviderOptionsV2
//#include <onnxruntime_c_api.h>


class Yolov8Onnx {
public:
	Yolov8Onnx() :_OrtMemoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPUOutput)) {};
	~Yolov8Onnx() {};// delete _OrtMemoryInfo;


public:
	/** \brief Read onnx-model
	* \param[in] modelPath:onnx-model path
	* \param[in] isCuda:if true,use Ort-GPU,else run it on cpu.
	* \param[in] cudaID:if isCuda==true,run Ort-GPU on cudaID.
	* \param[in] warmUp:if isCuda==true,warm up GPU-model.
	*/
	bool ReadModel(const std::string& modelPath, bool isCuda = false, int cudaID = 0, bool warmUp = true);

	/** \brief  detect.
	* \param[in] srcImg:a 3-channels image.
	* \param[out] output:detection results of input image.
	*/
	bool OnnxDetect(cv::Mat& srcImg, std::vector<OutputSeg>& output);
	/** \brief  detect,batch size= _batchSize
	* \param[in] srcImg:A batch of images.
	* \param[out] output:detection results of input images.
	*/
	bool OnnxBatchDetect(std::vector<cv::Mat>& srcImg, std::vector<std::vector<OutputSeg>>& output);

private:

	template <typename T>
	T VectorProduct(const std::vector<T>& v)
	{
		return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
	};
	int Preprocessing(const std::vector<cv::Mat>& SrcImgs, std::vector<cv::Mat>& OutSrcImgs, std::vector<cv::Vec4d>& params);
#if(defined YOLO_P6 && YOLO_P6==true)
	//const float _netAnchors[4][6] = { { 19,27, 44,40, 38,94 },{ 96,68, 86,152, 180,137 },{ 140,301, 303,264, 238,542 },{ 436,615, 739,380, 925,792 } };
	const int _netWidth = 1280;  //ONNX图片输入宽度
	const int _netHeight = 1280; //ONNX图片输入高度
	const int _segWidth = 320;  //_segWidth=_netWidth/mask_ratio
	const int _segHeight = 320;
	const int _segChannels = 32;

#else
	//const float _netAnchors[3][6] = { { 10,13, 16,30, 33,23 },{ 30,61, 62,45, 59,119 },{ 116,90, 156,198, 373,326 } };
	const int _netWidth = 640;   //ONNX-net-input-width
	const int _netHeight = 640;  //ONNX-net-input-height
	const int _segWidth = 160;    //_segWidth=_netWidth/mask_ratio
	const int _segHeight = 160;
	const int _segChannels = 32;

#endif // YOLO_P6

	int _batchSize = 1;  //if multi-batch,set this
	bool _isDynamicShape = false;//onnx support dynamic shape


	float _classThreshold = 0.25;
	float _nmsThreshold = 0.45;
	float _maskThreshold = 0.5;


	//ONNXRUNTIME	
	Ort::Env _OrtEnv = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Yolov5-Seg");
	Ort::SessionOptions _OrtSessionOptions = Ort::SessionOptions();
	Ort::Session* _OrtSession = nullptr;
	Ort::MemoryInfo _OrtMemoryInfo;
#if ORT_API_VERSION < ORT_OLD_VISON
	char* _inputName, * _output_name0;
#else
	std::shared_ptr<char> _inputName, _output_name0;
#endif

	std::vector<char*> _inputNodeNames; //输入节点名
	std::vector<char*> _outputNodeNames;//输出节点名

	size_t _inputNodesNum = 0;        //输入节点数
	size_t _outputNodesNum = 0;       //输出节点数

	ONNXTensorElementDataType _inputNodeDataType; //数据类型
	ONNXTensorElementDataType _outputNodeDataType;
	std::vector<int64_t> _inputTensorShape; //输入张量shape

	std::vector<int64_t> _outputTensorShape;

public:
	std::vector<std::string> _className = {
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush"
	};



};