#pragma once
#include "yolov8_utils.h"
#include <iostream>
#include <memory>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

// #include <tensorrt_provider_factory.h>  //if use OrtTensorRTProviderOptionsV2
// #include <onnxruntime_c_api.h>

class Yolov8PoseOnnx {
public:
  Yolov8PoseOnnx()
      : _OrtMemoryInfo(
            Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                       OrtMemType::OrtMemTypeCPUOutput)) {};
  ~Yolov8PoseOnnx() {
    if (_OrtSession != nullptr)
      delete _OrtSession;

  }; // delete _OrtMemoryInfo;

public:
  /** \brief Read onnx-model
   * \param[in] modelPath:onnx-model path
   * \param[in] isCuda:if true,use Ort-GPU,else run it on cpu.
   * \param[in] cudaID:if isCuda==true,run Ort-GPU on cudaID.
   * \param[in] warmUp:if isCuda==true,warm up GPU-model.
   */
  bool ReadModel(const std::string &modelPath, bool isCuda = false,
                 int cudaID = 0, bool warmUp = true);

  /** \brief  detect.
   * \param[in] srcImg:a 3-channels image.
   * \param[out] output:detection results of input image.
   */
  bool OnnxDetect(cv::Mat &srcImg, std::vector<OutputParams> &output);
  /** \brief  detect,batch size= _batchSize
   * \param[in] srcImg:A batch of images.
   * \param[out] output:detection results of input images.
   */
  bool OnnxBatchDetect(std::vector<cv::Mat> &srcImg,
                       std::vector<std::vector<OutputParams>> &output);

private:
  template <typename T> T VectorProduct(const std::vector<T> &v) {
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
  };
  int Preprocessing(const std::vector<cv::Mat> &srcImgs,
                    std::vector<cv::Mat> &outSrcImgs,
                    std::vector<cv::Vec4d> &params);

  const int _netWidth = 640;  // ONNX-net-input-width
  const int _netHeight = 640; // ONNX-net-input-height

  int _batchSize = 1;           // if multi-batch,set this
  bool _isDynamicShape = false; // onnx support dynamic shape
  float _classThreshold = 0.25;
  float _nmsThreshold = 0.45;

  // ONNXRUNTIME
  Ort::Env _OrtEnv =
      Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Yolov8");
  Ort::SessionOptions _OrtSessionOptions = Ort::SessionOptions();
  Ort::Session *_OrtSession = nullptr;
  Ort::MemoryInfo _OrtMemoryInfo;
#if ORT_API_VERSION < ORT_OLD_VISON
  char *_inputName, *_output_name0;
#else
  std::shared_ptr<char> _inputName, _output_name0;
#endif

  std::vector<char *> _inputNodeNames;  // ����ڵ���
  std::vector<char *> _outputNodeNames; // ����ڵ���

  size_t _inputNodesNum = 0;  // ����ڵ���
  size_t _outputNodesNum = 0; // ����ڵ���

  ONNXTensorElementDataType _inputNodeDataType; // ��������
  ONNXTensorElementDataType _outputNodeDataType;
  std::vector<int64_t> _inputTensorShape; // ��������shape

  std::vector<int64_t> _outputTensorShape;

public:
  std::vector<std::string> _className = {"person"};
};