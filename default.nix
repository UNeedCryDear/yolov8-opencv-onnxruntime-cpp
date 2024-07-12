{ pkgs ? import <nixpkgs> { } }:
let
  onnxruntime-gpu = (pkgs.onnxruntime.override {
    cudaSupport = true;
    cudaPackages = pkgs.cudaPackages;
  });
  opencv-gtk = (pkgs.opencv.override {
    # 启用 GTK2，便于显示图像
    enableGtk2 = true;
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
      cudaPackages.cudnn
    ] ++ [ onnxruntime-gpu opencv-gtk ];

  # 传给 CMake 的配置参数，控制 liboqs 的功能
  cmakeFlags =
    [ 
    "-DOpenCV_DIR=${pkgs.opencv4}"
    "-DONNXRUNTIME_DIR=${onnxruntime-gpu}"
    ];

  postInstall = ''
    chmod +x $out/bin/YOLOv8
    # mkdir -p $out/bin
    # install -m755 YOLOv8 $out/bin
   '';
}
