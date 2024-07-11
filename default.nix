{ pkgs ? import <nixpkgs> { } }:
let
  onnxruntime-gpu = (pkgs.onnxruntime.override {
    cudaSupport = true;
    cudaPackages = pkgs.cudaPackages;
  });
in pkgs.stdenv.mkDerivation rec {
  pname = "yolov8-detect";
  version = "1.0";

  src = ./.;

  enableParallelBuilding = true;
  # 此选项禁用了对 CMake 软件包的一些自动修正
  # dontFixCmake = true;

  nativeBuildInputs = with pkgs;
    [
      cmake
      opencv4
      cudaPackages.cudnn
      # cudaPackages.tensorrt
    ] ++ [ onnxruntime-gpu ];

  # 传给 CMake 的配置参数，控制 liboqs 的功能
  cmakeFlags =
    [ "-DOpenCV_DIR=${pkgs.opencv4}" "-DONNXRUNTIME_DIR=${onnxruntime-gpu}" ];

  # postInstall = ''
  #   mkdir -p $out/bin
  #   install -m755 bin/* "$out/bin"
  #  '';
}
