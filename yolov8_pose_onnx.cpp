#include "yolov8_pose_onnx.h"
// using namespace std;
// using namespace cv;
// using namespace cv::dnn;
using namespace Ort;

bool Yolov8PoseOnnx::ReadModel(const std::string &modelPath, bool isCuda,
                               int cudaID, bool warmUp) {
  if (_batchSize < 1)
    _batchSize = 1;
  try {
    if (!CheckModelPath(modelPath))
      return false;
    std::vector<std::string> available_providers = GetAvailableProviders();
    auto cuda_available =
        std::find(available_providers.begin(), available_providers.end(),
                  "CUDAExecutionProvider");

    if (isCuda && (cuda_available == available_providers.end())) {
      std::cout << "Your ORT build without GPU. Change to CPU." << std::endl;
      std::cout << "************* Infer model on CPU! *************"
                << std::endl;
    } else if (isCuda && (cuda_available != available_providers.end())) {
      std::cout << "************* Infer model on GPU! *************"
                << std::endl;
#if ORT_API_VERSION < ORT_OLD_VISON
      OrtCUDAProviderOptions cudaOption;
      cudaOption.device_id = cudaID;
      _OrtSessionOptions.AppendExecutionProvider_CUDA(cudaOption);
#else
      OrtStatus *status = OrtSessionOptionsAppendExecutionProvider_CUDA(
          _OrtSessionOptions, cudaID);
#endif
    } else {
      std::cout << "************* Infer model on CPU! *************"
                << std::endl;
    }
    //

    _OrtSessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
    std::wstring model_path(modelPath.begin(), modelPath.end());
    _OrtSession =
        new Ort::Session(_OrtEnv, model_path.c_str(), _OrtSessionOptions);
#else
    _OrtSession =
        new Ort::Session(_OrtEnv, modelPath.c_str(), _OrtSessionOptions);
#endif

    Ort::AllocatorWithDefaultOptions allocator;
    // init input
    _inputNodesNum = _OrtSession->GetInputCount();
#if ORT_API_VERSION < ORT_OLD_VISON
    _inputName = _OrtSession->GetInputName(0, allocator);
    _inputNodeNames.push_back(_inputName);
#else
    _inputName = std::move(_OrtSession->GetInputNameAllocated(0, allocator));
    _inputNodeNames.push_back(_inputName.get());
#endif
    // std::cout << _inputNodeNames[0] << std::endl;
    Ort::TypeInfo inputTypeInfo = _OrtSession->GetInputTypeInfo(0);
    auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
    _inputNodeDataType = input_tensor_info.GetElementType();
    _inputTensorShape = input_tensor_info.GetShape();

    if (_inputTensorShape[0] == -1) {
      _isDynamicShape = true;
      _inputTensorShape[0] = _batchSize;
    }
    if (_inputTensorShape[2] == -1 || _inputTensorShape[3] == -1) {
      _isDynamicShape = true;
      _inputTensorShape[2] = _netHeight;
      _inputTensorShape[3] = _netWidth;
    }
    // init output
    _outputNodesNum = _OrtSession->GetOutputCount();
#if ORT_API_VERSION < ORT_OLD_VISON
    _output_name0 = _OrtSession->GetOutputName(0, allocator);
    _outputNodeNames.push_back(_output_name0);
#else
    _output_name0 =
        std::move(_OrtSession->GetOutputNameAllocated(0, allocator));
    _outputNodeNames.push_back(_output_name0.get());
#endif
    Ort::TypeInfo type_info_output0(nullptr);
    type_info_output0 = _OrtSession->GetOutputTypeInfo(0); // output0

    auto tensor_info_output0 = type_info_output0.GetTensorTypeAndShapeInfo();
    _outputNodeDataType = tensor_info_output0.GetElementType();
    _outputTensorShape = tensor_info_output0.GetShape();

    //_outputMaskNodeDataType = tensor_info_output1.GetElementType(); //the same
    //as output0 _outputMaskTensorShape = tensor_info_output1.GetShape(); if
    // (_outputTensorShape[0] == -1)
    //{
    //	_outputTensorShape[0] = _batchSize;
    //	_outputMaskTensorShape[0] = _batchSize;
    //}
    // if (_outputMaskTensorShape[2] == -1) {
    //	//size_t ouput_rows = 0;
    //	//for (int i = 0; i < _strideSize; ++i) {
    //	//	ouput_rows += 3 * (_netWidth / _netStride[i]) * _netHeight /
    //_netStride[i];
    //	//}
    //	//_outputTensorShape[1] = ouput_rows;

    //	_outputMaskTensorShape[2] = _segHeight;
    //	_outputMaskTensorShape[3] = _segWidth;
    //}

    // warm up
    if (isCuda && warmUp) {
      // draw run
      std::cout << "Start warming up" << std::endl;
      size_t input_tensor_length = VectorProduct(_inputTensorShape);
      float *temp = new float[input_tensor_length];
      std::vector<Ort::Value> input_tensors;
      std::vector<Ort::Value> output_tensors;
      input_tensors.push_back(Ort::Value::CreateTensor<float>(
          _OrtMemoryInfo, temp, input_tensor_length, _inputTensorShape.data(),
          _inputTensorShape.size()));
      for (int i = 0; i < 3; ++i) {
        output_tensors =
            _OrtSession->Run(Ort::RunOptions{nullptr}, _inputNodeNames.data(),
                             input_tensors.data(), _inputNodeNames.size(),
                             _outputNodeNames.data(), _outputNodeNames.size());
      }

      delete[] temp;
    }
  } catch (const std::exception &) {
    return false;
  }
  return true;
}

int Yolov8PoseOnnx::Preprocessing(const std::vector<cv::Mat> &srcImgs,
                                  std::vector<cv::Mat> &outSrcImgs,
                                  std::vector<cv::Vec4d> &params) {
  outSrcImgs.clear();
  cv::Size input_size = cv::Size(_netWidth, _netHeight);
  for (int i = 0; i < srcImgs.size(); ++i) {
    cv::Mat temp_img = srcImgs[i];
    cv::Vec4d temp_param = {1, 1, 0, 0};
    if (temp_img.size() != input_size) {
      cv::Mat borderImg;
      LetterBox(temp_img, borderImg, temp_param, input_size, false, false, true,
                32);
      // std::cout << borderImg.size() << std::endl;
      outSrcImgs.push_back(borderImg);
      params.push_back(temp_param);
    } else {
      outSrcImgs.push_back(temp_img);
      params.push_back(temp_param);
    }
  }

  int lack_num = _batchSize - srcImgs.size();
  if (lack_num > 0) {
    for (int i = 0; i < lack_num; ++i) {
      cv::Mat temp_img = cv::Mat::zeros(input_size, CV_8UC3);
      cv::Vec4d temp_param = {1, 1, 0, 0};
      outSrcImgs.push_back(temp_img);
      params.push_back(temp_param);
    }
  }
  return 0;
}
bool Yolov8PoseOnnx::OnnxDetect(cv::Mat &srcImg,
                                std::vector<OutputParams> &output) {
  std::vector<cv::Mat> input_data = {srcImg};
  std::vector<std::vector<OutputParams>> tenp_output;
  if (OnnxBatchDetect(input_data, tenp_output)) {
    output = tenp_output[0];
    return true;
  } else
    return false;
}
bool Yolov8PoseOnnx::OnnxBatchDetect(
    std::vector<cv::Mat> &srcImgs,
    std::vector<std::vector<OutputParams>> &output) {
  std::vector<cv::Vec4d> params;
  std::vector<cv::Mat> input_images;
  cv::Size input_size(_netWidth, _netHeight);
  // preprocessing
  Preprocessing(srcImgs, input_images, params);
  cv::Mat blob = cv::dnn::blobFromImages(input_images, 1 / 255.0, input_size,
                                         cv::Scalar(0, 0, 0), true, false);

  int64_t input_tensor_length = VectorProduct(_inputTensorShape);
  std::vector<Ort::Value> input_tensors;
  std::vector<Ort::Value> output_tensors;
  input_tensors.push_back(Ort::Value::CreateTensor<float>(
      _OrtMemoryInfo, (float *)blob.data, input_tensor_length,
      _inputTensorShape.data(), _inputTensorShape.size()));

  output_tensors = _OrtSession->Run(
      Ort::RunOptions{nullptr}, _inputNodeNames.data(), input_tensors.data(),
      _inputNodeNames.size(), _outputNodeNames.data(), _outputNodeNames.size());
  // post-process
  float *all_data = output_tensors[0].GetTensorMutableData<float>();
  _outputTensorShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
  int net_width = _outputTensorShape[1];

  int key_point_length = net_width - 5;
  int key_point_num = 17; //_bodyKeyPoints.size(), shape[x, y, confidence]
  if (key_point_num * 3 != key_point_length) {
    std::cout << "Pose should be shape [x, y, confidence] with 17-points"
              << std::endl;
    return false;
  }
  int64_t one_output_length =
      VectorProduct(_outputTensorShape) / _outputTensorShape[0];
  for (int img_index = 0; img_index < srcImgs.size(); ++img_index) {
    cv::Mat output0 = cv::Mat(cv::Size((int)_outputTensorShape[2],
                                       (int)_outputTensorShape[1]),
                              CV_32F, all_data)
                          .t(); //[bs,116,8400]=>[bs,8400,116]
    all_data += one_output_length;
    float *pdata = (float *)output0.data;
    int rows = output0.rows;
    std::vector<int> class_ids;     // ���id����
    std::vector<float> confidences; // ���ÿ��id��Ӧ���Ŷ�����
    std::vector<cv::Rect> boxes;    // ÿ��id���ο�
    std::vector<std::vector<PoseKeyPoint>> pose_key_points; // ����kpt

    for (int r = 0; r < rows; ++r) {
      float max_class_socre = pdata[4];
      if (max_class_socre >= _classThreshold) {
        // rect [x,y,w,h]
        float x = (pdata[0] - params[img_index][2]) / params[img_index][0]; // x
        float y = (pdata[1] - params[img_index][3]) / params[img_index][1]; // y
        float w = pdata[2] / params[img_index][0];                          // w
        float h = pdata[3] / params[img_index][1];                          // h
        int left = MAX(int(x - 0.5 * w + 0.5), 0);
        int top = MAX(int(y - 0.5 * h + 0.5), 0);
        class_ids.push_back(0);
        confidences.push_back(max_class_socre);
        boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
        std::vector<PoseKeyPoint> temp_kpts;
        for (int kpt = 0; kpt < key_point_length; kpt += 3) {
          PoseKeyPoint temp_kp;
          temp_kp.x =
              (pdata[5 + kpt] - params[img_index][2]) / params[img_index][0];
          temp_kp.y =
              (pdata[6 + kpt] - params[img_index][3]) / params[img_index][1];
          temp_kp.confidence = pdata[7 + kpt];
          temp_kpts.push_back(temp_kp);
        }
        pose_key_points.push_back(temp_kpts);
      }
      pdata += net_width; // ��һ��
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold,
                      nms_result);
    std::vector<std::vector<float>> temp_mask_proposals;
    cv::Rect holeImgRect(0, 0, srcImgs[img_index].cols,
                         srcImgs[img_index].rows);
    std::vector<OutputParams> temp_output;
    for (int i = 0; i < nms_result.size(); ++i) {
      int idx = nms_result[i];
      OutputParams result;
      result.id = class_ids[idx];
      result.confidence = confidences[idx];
      result.box = boxes[idx] & holeImgRect;
      result.keyPoints = pose_key_points[idx];
      temp_output.push_back(result);
    }
    output.push_back(temp_output);
  }

  if (output.size())
    return true;
  else
    return false;
}