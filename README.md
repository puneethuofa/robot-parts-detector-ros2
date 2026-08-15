# Real-Time Industrial Parts Detector — ROS2 + TensorRT

End-to-end pipeline: train a YOLOv8 detector on an industrial-parts/bin-picking
dataset using SLURM on a university HPC cluster, export to ONNX, optimize with
TensorRT, benchmark before vs. after, then serve the optimized model as a
ROS2 node that subscribes to a camera topic and publishes detections.

```
Camera topic ──▶ ROS2 node (rclpy) ──▶ TensorRT engine ──▶ Detection2DArray topic
                                              ▲
                                   built from ONNX export of a
                                   YOLOv8 model trained on HPC
```

## Repo layout

```
train/
  download_dataset.py     # pulls a Roboflow industrial-parts dataset
  train.py                # ultralytics YOLOv8 training entrypoint
  slurm_train.sbatch      # SLURM job script for HPC GPU nodes
  export_onnx.py          # PyTorch -> ONNX export

optimize/
  build_trt_engine.py     # ONNX -> TensorRT engine (FP32/FP16/INT8)
  benchmark.py            # latency/FPS/GPU-util benchmark, before vs after

ros2_ws/src/part_detector/
  part_detector/
    trt_inference.py      # TensorRT engine wrapper (pycuda-based)
    detector_node.py      # rclpy node: image topic in, Detection2DArray out
    utils.py              # preprocessing / NMS / letterbox helpers
  launch/detector.launch.py
  package.xml / setup.py / setup.cfg

docker/Dockerfile          # ROS2 Humble + TensorRT runtime image
docs/benchmark_template.md # fill this in with your own numbers after running
```

## 1. Train (on HPC)

```bash
cd train
python download_dataset.py --dataset roboflow-universe/industrial-parts --out ./data
sbatch slurm_train.sbatch      # edit account/partition/module lines for your cluster first
```

`slurm_train.sbatch` requests 1 GPU, loads a CUDA/cuDNN module, activates a
venv, and calls `train.py`, which is a thin wrapper around
`ultralytics.YOLO(...).train(...)` with the dataset's `data.yaml`.

## 2. Export to ONNX

```bash
python train/export_onnx.py --weights runs/detect/train/weights/best.pt \
    --imgsz 640 --out weights/detector.onnx
```

## 3. Build the TensorRT engine

```bash
python optimize/build_trt_engine.py --onnx weights/detector.onnx \
    --engine weights/detector_fp16.engine --precision fp16
```

Supports `fp32`, `fp16`, and `int8` (INT8 needs a small calibration image
folder — see `--calib-dir`).

## 4. Benchmark before vs. after

```bash
python optimize/benchmark.py --onnx weights/detector.onnx \
    --engine weights/detector_fp16.engine --imgsz 640 --iters 500
```

Numbers depend on your GPU — fill the actual run output into
`docs/benchmark_template.md` for your README/resume.

## 5. Run the ROS2 node

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch part_detector detector.launch.py \
    engine_path:=/absolute/path/weights/detector_fp16.engine \
    image_topic:=/camera/image_raw
```

Test against a rosbag instead of a live camera:

```bash
ros2 bag play my_bin_picking.bag --topics /camera/image_raw
ros2 topic echo /part_detector/detections
```

## Requirements

- ROS2 Humble (or newer)
- Python 3.10+
- `ultralytics`, `onnx`, `onnxruntime-gpu`, `tensorrt`, `pycuda`, `opencv-python`
- CUDA-capable GPU for training/TensorRT steps (HPC cluster or local)

See `requirements.txt` for pinned versions.

## Notes / honest scope

- This repo is a complete, runnable scaffold: correct ROS2 package structure,
  a working TensorRT inference wrapper, and training/export/benchmark
  scripts. It ships without pretrained weights or an executed benchmark —
  those depend on your HPC allocation and dataset license, and are meant to
  be produced by you running the pipeline end-to-end and dropping your real
  numbers into `docs/benchmark_template.md`.
- The ROS2 node is Python (`rclpy`). A C++ (`rclcpp`) version of the same
  node is a natural follow-up if that's a gap you're also trying to close.
