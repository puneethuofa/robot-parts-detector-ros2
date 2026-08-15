#!/usr/bin/env python3
"""
Build a TensorRT engine from an ONNX model, with FP32 / FP16 / INT8 support.

Usage:
    python build_trt_engine.py --onnx ../weights/detector.onnx \
        --engine ../weights/detector_fp16.engine --precision fp16

    # INT8 (needs a folder of representative calibration images):
    python build_trt_engine.py --onnx ../weights/detector.onnx \
        --engine ../weights/detector_int8.engine --precision int8 \
        --calib-dir ./calib_images --imgsz 640
"""
import argparse
import glob
import os

import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    """Minimal INT8 calibrator: reads a folder of images, feeds them through
    the same preprocessing used at inference time (see part_detector/utils.py)."""

    def __init__(self, calib_dir, imgsz, batch_size=8, cache_file="calibration.cache"):
        super().__init__()
        import cv2  # local import so this file only requires cv2 for INT8 path

        self.cv2 = cv2
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.image_paths = sorted(
            glob.glob(os.path.join(calib_dir, "*.jpg"))
            + glob.glob(os.path.join(calib_dir, "*.png"))
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No calibration images found in {calib_dir}")
        self.index = 0

        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        self.cuda = cuda
        nbytes = batch_size * 3 * imgsz * imgsz * np.dtype(np.float32).itemsize
        self.device_input = cuda.mem_alloc(nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.index + self.batch_size > len(self.image_paths):
            return None

        batch_imgs = []
        for path in self.image_paths[self.index : self.index + self.batch_size]:
            img = self.cv2.imread(path)
            img = self.cv2.resize(img, (self.imgsz, self.imgsz))
            img = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            batch_imgs.append(img)

        batch = np.ascontiguousarray(np.stack(batch_imgs))
        self.cuda.memcpy_htod(self.device_input, batch)
        self.index += self.batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_engine(onnx_path, engine_path, precision, imgsz, calib_dir=None, workspace_gb=4):
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"Failed to parse ONNX file: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("Warning: platform reports no fast FP16 support, building anyway.")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            print("Warning: platform reports no fast INT8 support, building anyway.")
        if calib_dir is None:
            raise ValueError("--calib-dir is required for --precision int8")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = ImageCalibrator(calib_dir, imgsz)
    elif precision != "fp32":
        raise ValueError(f"Unknown precision: {precision}")

    print(f"Building TensorRT engine ({precision})... this can take a few minutes.")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed.")

    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"Engine written to: {engine_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--calib-dir", default=None, help="Required for --precision int8")
    parser.add_argument("--workspace-gb", type=int, default=4)
    args = parser.parse_args()

    build_engine(
        args.onnx, args.engine, args.precision, args.imgsz,
        calib_dir=args.calib_dir, workspace_gb=args.workspace_gb,
    )


if __name__ == "__main__":
    main()
