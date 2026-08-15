#!/usr/bin/env python3
"""
Benchmark inference latency / FPS / GPU utilization across three backends:
PyTorch (.pt), ONNX Runtime (CUDAExecutionProvider), and TensorRT (.engine).

This is the "before vs. after optimization" evidence for the pipeline: run
this once with just --weights (PyTorch baseline + ONNX Runtime), then again
after building the engine to add the TensorRT row.

Usage:
    python benchmark.py --weights ../train/runs/detect/parts_v1/weights/best.pt \
        --onnx ../weights/detector.onnx --engine ../weights/detector_fp16.engine \
        --imgsz 640 --iters 500 --warmup 50
"""
import argparse
import json
import time
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class BenchResult:
    backend: str
    avg_latency_ms: float
    p95_latency_ms: float
    fps: float
    gpu_util_pct: float | None


def _sample_gpu_util():
    """Best-effort GPU utilization sample via nvidia-smi (no extra deps)."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            timeout=2,
        )
        return float(out.decode().strip().splitlines()[0])
    except Exception:
        return None


def _random_batch(imgsz, n=1):
    return np.random.rand(n, 3, imgsz, imgsz).astype(np.float32)


def bench_pytorch(weights, imgsz, iters, warmup):
    from ultralytics import YOLO

    model = YOLO(weights)
    dummy = _random_batch(imgsz)

    for _ in range(warmup):
        model.predict(dummy, imgsz=imgsz, verbose=False)

    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(dummy, imgsz=imgsz, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    return _summarize("PyTorch (.pt)", latencies)


def bench_onnxruntime(onnx_path, imgsz, iters, warmup):
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    dummy = _random_batch(imgsz)

    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    return _summarize("ONNX Runtime (CUDA)", latencies)


def bench_tensorrt(engine_path, imgsz, iters, warmup):
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ros2_ws", "src", "part_detector"))
    from part_detector.trt_inference import TRTEngine  # noqa: E402

    engine = TRTEngine(engine_path)
    dummy = _random_batch(imgsz)[0]  # TRTEngine takes CHW single image

    for _ in range(warmup):
        engine.infer(dummy)

    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        engine.infer(dummy)
        latencies.append((time.perf_counter() - t0) * 1000)

    engine.close()
    return _summarize("TensorRT", latencies)


def _summarize(name, latencies_ms):
    arr = np.array(latencies_ms)
    avg = float(arr.mean())
    p95 = float(np.percentile(arr, 95))
    fps = 1000.0 / avg if avg > 0 else 0.0
    gpu_util = _sample_gpu_util()
    return BenchResult(name, round(avg, 2), round(p95, 2), round(fps, 1), gpu_util)


def print_table(results):
    header = f"{'Backend':<24}{'Avg latency (ms)':>18}{'P95 (ms)':>12}{'FPS':>10}{'GPU util (%)':>14}"
    print(header)
    print("-" * len(header))
    for r in results:
        gpu_str = f"{r.gpu_util_pct:.0f}" if r.gpu_util_pct is not None else "n/a"
        print(f"{r.backend:<24}{r.avg_latency_ms:>18.2f}{r.p95_latency_ms:>12.2f}{r.fps:>10.1f}{gpu_str:>14}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=None, help="PyTorch .pt weights (optional)")
    parser.add_argument("--onnx", default=None, help="ONNX model path (optional)")
    parser.add_argument("--engine", default=None, help="TensorRT .engine path (optional)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--json-out", default=None, help="Optional path to dump results as JSON")
    args = parser.parse_args()

    results = []
    if args.weights:
        print("Benchmarking PyTorch baseline...")
        results.append(bench_pytorch(args.weights, args.imgsz, args.iters, args.warmup))
    if args.onnx:
        print("Benchmarking ONNX Runtime...")
        results.append(bench_onnxruntime(args.onnx, args.imgsz, args.iters, args.warmup))
    if args.engine:
        print("Benchmarking TensorRT engine...")
        results.append(bench_tensorrt(args.engine, args.imgsz, args.iters, args.warmup))

    if not results:
        raise SystemExit("Provide at least one of --weights / --onnx / --engine")

    print()
    print_table(results)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults written to {args.json_out}")


if __name__ == "__main__":
    main()
