#include "yolov8_seg_onnx.h"
using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace Ort;

bool Yolov8SegOnnx::ReadModel(const std::string& modelPath, bool isCuda, int cudaID, bool warmUp) {
	if (_batchSize < 1) _batchSize = 1;
	try
	{
		std::vector<std::string> available_providers = GetAvailableProviders();
		auto cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider");
	
		if (isCuda && (cuda_available == available_providers.end()))
		{
			std::cout << "Your ORT build without GPU. Change to CPU." << std::endl;
			std::cout << "************* Infer model on CPU! *************" << std::endl;
		}
		else if (isCuda && (cuda_available != available_providers.end()))
		{
			std::cout << "************* Infer model on GPU! *************" << std::endl;
#if ORT_API_VERSION < ORT_OLD_VISON
			OrtCUDAProviderOptions cudaOption;
			cudaOption.device_id = cudaID;
			_OrtSessionOptions.AppendExecutionProvider_CUDA(cudaOption);
#else
			OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(_OrtSessionOptions, cudaID);
#endif
		}
		else
		{
			std::cout << "************* Infer model on CPU! *************" << std::endl;
		}
		//
		_OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
		std::wstring model_path(modelPath.begin(), modelPath.end());
		_OrtSession = new Ort::Session(_OrtEnv, model_path.c_str(), _OrtSessionOptions);
#else
		_OrtSession = new Ort::Session(_OrtEnv, modelPath.c_str(), _OrtSessionOptions);
#endif

		Ort::AllocatorWithDefaultOptions allocator;
		//init input
		_inputNodesNum = _OrtSession->GetInputCount();
#if ORT_API_VERSION < ORT_OLD_VISON
		_inputName = _OrtSession->GetInputName(0, allocator);
		_inputNodeNames.push_back(_inputName);
#else
		_inputName = std::move(_OrtSession->GetInputNameAllocated(0, allocator));
		_inputNodeNames.push_back(_inputName.get());
#endif
	
		Ort::TypeInfo inputTypeInfo = _OrtSession->GetInputTypeInfo(0);
		auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
		_inputNodeDataType = input_tensor_info.GetElementType();
		_inputTensorShape = input_tensor_info.GetShape();

		if (_inputTensorShape[0] == -1)
		{
			_isDynamicShape = true;
			_inputTensorShape[0] = _batchSize;

		}
		if (_inputTensorShape[2] == -1 || _inputTensorShape[3] == -1) {
			_isDynamicShape = true;
			_inputTensorShape[2] = _netHeight;
			_inputTensorShape[3] = _netWidth;
		}
		//init output
		_outputNodesNum = _OrtSession->GetOutputCount();
		if (_outputNodesNum != 2) {
			cout << "This model has " << _outputNodesNum << "output, which is not a segmentation model.Please check your model name or path!" << endl;
			return false;
		}
#if ORT_API_VERSION < ORT_OLD_VISON
		_output_name0 = _OrtSession->GetOutputName(0, allocator);
		_output_name1 = _OrtSession->GetOutputName(1, allocator);
#else
		_output_name0 = std::move(_OrtSession->GetOutputNameAllocated(0, allocator));
		_output_name1 = std::move(_OrtSession->GetOutputNameAllocated(1, allocator));
#endif
		Ort::TypeInfo type_info_output0(nullptr);
		Ort::TypeInfo type_info_output1(nullptr);
		bool flag = false;
#if ORT_API_VERSION < ORT_OLD_VISON
		flag = strcmp(_output_name0, _output_name1) < 0;
#else
		flag = strcmp(_output_name0.get(), _output_name1.get()) < 0;
#endif
		if (flag)  //make sure "output0" is in front of  "output1"
		{
			type_info_output0 = _OrtSession->GetOutputTypeInfo(0);  //output0
			type_info_output1 = _OrtSession->GetOutputTypeInfo(1);  //output1
#if ORT_API_VERSION < ORT_OLD_VISON
			_outputNodeNames.push_back(_output_name0);
			_outputNodeNames.push_back(_output_name1);
#else
			_outputNodeNames.push_back(_output_name0.get());
			_outputNodeNames.push_back(_output_name1.get());
#endif
		
		}
		else {
			type_info_output0 = _OrtSession->GetOutputTypeInfo(1);  //output0
			type_info_output1 = _OrtSession->GetOutputTypeInfo(0);  //output1
#if ORT_API_VERSION < ORT_OLD_VISON
			_outputNodeNames.push_back(_output_name1);
			_outputNodeNames.push_back(_output_name0);
#else
			_outputNodeNames.push_back(_output_name1.get());
			_outputNodeNames.push_back(_output_name0.get());
#endif
		}

		auto tensor_info_output0 = type_info_output0.GetTensorTypeAndShapeInfo();
		_outputNodeDataType = tensor_info_output0.GetElementType();
		_outputTensorShape = tensor_info_output0.GetShape();
		auto tensor_info_output1 = type_info_output1.GetTensorTypeAndShapeInfo();
		//_outputMaskNodeDataType = tensor_info_output1.GetElementType(); //the same as output0
		//_outputMaskTensorShape = tensor_info_output1.GetShape();
		//if (_outputTensorShape[0] == -1)
		//{
		//	_outputTensorShape[0] = _batchSize;
		//	_outputMaskTensorShape[0] = _batchSize;
		//}
		//if (_outputMaskTensorShape[2] == -1) {
		//	//size_t ouput_rows = 0;
		//	//for (int i = 0; i < _strideSize; ++i) {
		//	//	ouput_rows += 3 * (_netWidth / _netStride[i]) * _netHeight / _netStride[i];
		//	//}
		//	//_outputTensorShape[1] = ouput_rows;

		//	_outputMaskTensorShape[2] = _segHeight;
		//	_outputMaskTensorShape[3] = _segWidth;
		//}
		//warm up
		if (isCuda && warmUp) {
			//draw run
			cout << "Start warming up" << endl;
			size_t input_tensor_length = VectorProduct(_inputTensorShape);
			float* temp = new float[input_tensor_length];
			std::vector<Ort::Value> input_tensors;
			std::vector<Ort::Value> output_tensors;
			input_tensors.push_back(Ort::Value::CreateTensor<float>(
				_OrtMemoryInfo, temp, input_tensor_length, _inputTensorShape.data(),
				_inputTensorShape.size()));
			for (int i = 0; i < 3; ++i) {
				output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
					_inputNodeNames.data(),
					input_tensors.data(),
					_inputNodeNames.size(),
					_outputNodeNames.data(),
					_outputNodeNames.size());
			}

			delete[]temp;
		}
	}
	catch (const std::exception&) {
		return false;
	}

}

int Yolov8SegOnnx::Preprocessing(const std::vector<cv::Mat>& srcImgs, std::vector<cv::Mat>& outSrcImgs, std::vector<cv::Vec4d>& params) {
	outSrcImgs.clear();
	Size input_size = Size(_netWidth, _netHeight);
	for (int i = 0; i < srcImgs.size(); ++i) {
		Mat temp_img = srcImgs[i];
		Vec4d temp_param = { 1,1,0,0 };
		if (temp_img.size() != input_size) {
			Mat borderImg;
			LetterBox(temp_img, borderImg, temp_param, input_size, false, false, true, 32);
			//cout << borderImg.size() << endl;
			outSrcImgs.push_back(borderImg);
			params.push_back(temp_param);
		}
		else {
			outSrcImgs.push_back(temp_img);
			params.push_back(temp_param);
		}
	}

	int lack_num = srcImgs.size() % _batchSize;
	if (lack_num != 0) {
		for (int i = 0; i < lack_num; ++i) {
			Mat temp_img = Mat::zeros(input_size, CV_8UC3);
			Vec4d temp_param = { 1,1,0,0 };
			outSrcImgs.push_back(temp_img);
			params.push_back(temp_param);
		}
	}
	return 0;

}
bool Yolov8SegOnnx::OnnxDetect(cv::Mat& srcImg, std::vector<OutputSeg>& output) {
	std::vector<cv::Mat> input_data = { srcImg };
	std::vector<std::vector<OutputSeg>> tenp_output;
	if (OnnxBatchDetect(input_data, tenp_output)) {
		output = tenp_output[0];
		return true;
	}
	else return false;
}
bool Yolov8SegOnnx::OnnxBatchDetect(std::vector<cv::Mat>& srcImgs, std::vector<std::vector<OutputSeg>>& output) {
	vector<Vec4d> params;
	vector<Mat> input_images;
	cv::Size input_size(_netWidth, _netHeight);
	//preprocessing
	Preprocessing(srcImgs, input_images, params);
	cv::Mat blob = cv::dnn::blobFromImages(input_images, 1 / 255.0, input_size, Scalar(0, 0, 0), true, false);

	int64_t input_tensor_length = VectorProduct(_inputTensorShape);
	std::vector<Ort::Value> input_tensors;
	std::vector<Ort::Value> output_tensors;
	input_tensors.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo, (float*)blob.data, input_tensor_length, _inputTensorShape.data(), _inputTensorShape.size()));

	output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
		_inputNodeNames.data(),
		input_tensors.data(),
		_inputNodeNames.size(),
		_outputNodeNames.data(),
		_outputNodeNames.size()
	);

	//post-process

	int net_width = _className.size() + 4 + _segChannels;
	float* all_data = output_tensors[0].GetTensorMutableData<float>();
	_outputTensorShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
	_outputMaskTensorShape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
	vector<int> mask_protos_shape = { 1,(int)_outputMaskTensorShape[1],(int)_outputMaskTensorShape[2],(int)_outputMaskTensorShape[3] };
	int mask_protos_length = VectorProduct(mask_protos_shape);
	int64_t one_output_length = VectorProduct(_outputTensorShape) / _outputTensorShape[0];
	for (int img_index = 0; img_index < srcImgs.size(); ++img_index) {
		Mat output0 = Mat(Size((int)_outputTensorShape[2], (int)_outputTensorShape[1]), CV_32F, all_data).t();  //[bs,116,8400]=>[bs,8400,116]
		all_data += one_output_length;
		float* pdata = (float*)output0.data;
		int rows = output0.rows;
		std::vector<int> class_ids;//结果id数组
		std::vector<float> confidences;//结果每个id对应置信度数组
		std::vector<cv::Rect> boxes;//每个id矩形框
		std::vector<vector<float>> picked_proposals;  //output0[:,:, 5 + _className.size():net_width]===> for mask
		for (int r = 0; r < rows; ++r) {    //stride
				cv::Mat scores(1, _className.size(), CV_32F, pdata +4);
				Point classIdPoint;
				double max_class_socre;
				minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
				max_class_socre = (float)max_class_socre;
				if (max_class_socre >= _classThreshold) {
					vector<float> temp_proto(pdata + 4 + _className.size(), pdata + net_width);
					picked_proposals.push_back(temp_proto);
					//rect [x,y,w,h]
					float x = (pdata[0] - params[img_index][2]) / params[img_index][0];  //x
					float y = (pdata[1] - params[img_index][3]) / params[img_index][1];  //y
					float w = pdata[2] / params[img_index][0];  //w
					float h = pdata[3] / params[img_index][1];  //h
					int left = MAX(int(x - 0.5 * w + 0.5), 0);
					int top = MAX(int(y - 0.5 * h + 0.5), 0);
					class_ids.push_back(classIdPoint.x);
					confidences.push_back(max_class_socre );
					boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
				}
				pdata += net_width;//下一行
		}

		vector<int> nms_result;
		cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);
		std::vector<vector<float>> temp_mask_proposals;
		cv::Rect holeImgRect(0, 0, srcImgs[img_index].cols, srcImgs[img_index].rows);
		std::vector<OutputSeg> temp_output;
		for (int i = 0; i < nms_result.size(); ++i) {
			int idx = nms_result[i];
			OutputSeg result;
			result.id = class_ids[idx];
			result.confidence = confidences[idx];
			result.box = boxes[idx] & holeImgRect;
			temp_mask_proposals.push_back(picked_proposals[idx]);
			temp_output.push_back(result);
		}

		MaskParams mask_params;
		mask_params.params = params[img_index];
		mask_params.srcImgShape = srcImgs[img_index].size();
		Mat mask_protos = Mat(mask_protos_shape, CV_32F, output_tensors[1].GetTensorMutableData<float>() + img_index * mask_protos_length);
		for (int i = 0; i < temp_mask_proposals.size(); ++i) {
			GetMask2(Mat(temp_mask_proposals[i]).t(), mask_protos, temp_output[i], mask_params);
		}

		//******************** ****************
		// 老版本的方案，如果上面在开启我注释的部分之后还一直报错，建议使用这个。
		// If the GetMask2() still reports errors , it is recommended to use GetMask().
		// Mat mask_proposals;
		//for (int i = 0; i < temp_mask_proposals.size(); ++i)
		//	mask_proposals.push_back(Mat(temp_mask_proposals[i]).t());
		//GetMask(mask_proposals, mask_protos, output, mask_params);
		//*****************************************************/
		output.push_back(temp_output);

	}

	if (output.size())
		return true;
	else
		return false;
}