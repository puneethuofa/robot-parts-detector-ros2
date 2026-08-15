#!/usr/bin/env python3
"""
Export a trained YOLOv8 .pt checkpoint to ONNX, then simplify the graph.

Usage:
    python export_onnx.py --weights runs/detect/parts_v1/weights/best.pt \
        --imgsz 640 --out ../weights/detector.onnx
"""
import argparse
import os
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", required=True, help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic batch axis")
    parser.add_argument("--simplify", action="store_true", default=True)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    exported_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if os.path.abspath(str(exported_path)) != os.path.abspath(args.out):
        shutil.move(str(exported_path), args.out)

    print(f"ONNX model written to: {args.out}")


if __name__ == "__main__":
    main()
