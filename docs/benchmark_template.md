# Benchmark Results

Fill this in after running `optimize/benchmark.py` on your own trained
weights and TensorRT engine. Include your GPU model — these numbers are
GPU-dependent and a reviewer will expect to see the hardware named.

**Hardware:** _e.g. NVIDIA A100 40GB / UofA HPC ElGato-GPU node_
**Dataset:** _dataset name, image count, number of classes_
**Model:** _e.g. YOLOv8s, trained N epochs_
**Image size:** _e.g. 640x640_

| Backend              | Avg latency (ms) | P95 latency (ms) | FPS  | GPU util (%) |
|-----------------------|------------------:|-------------------:|-----:|-------------:|
| PyTorch (.pt)          |                    |                     |      |               |
| ONNX Runtime (CUDA)    |                    |                     |      |               |
| TensorRT FP16           |                    |                     |      |               |
| TensorRT INT8 (optional) |                  |                     |      |               |

**Accuracy check (make sure optimization didn't silently break the model):**

| Backend        | mAP50 | mAP50-95 |
|-----------------|------:|---------:|
| PyTorch baseline |       |          |
| TensorRT FP16     |       |          |

## One-paragraph summary (for a resume bullet or README)

> Example shape to fill in: "Optimized a YOLOv8s parts detector with
> TensorRT FP16, reducing per-frame inference latency from **X ms to Y ms**
> (**Z× speedup**) and raising throughput from **A FPS to B FPS** on an
> NVIDIA ___, with mAP50 held within 0.5 points of the PyTorch baseline —
> deployed as a ROS2 node processing a live/rosbag camera feed in real time."
