#!/usr/bin/env python3
"""
Download an industrial-parts / bin-picking object detection dataset from
Roboflow Universe in YOLO format.

Good candidate datasets on Roboflow Universe (search these workspace/project
names in the Roboflow Universe UI to get the exact `workspace/project/version`
triple, since these change over time):

  - "mechanical-parts-detection"      (bolts, nuts, gears, industrial hardware)
  - "warehouse-objects"               (boxes, pallets, bins — general warehouse analog)
  - "bin-picking"                     (cluttered bin-picking scenes, closest to
                                        surface-prep / robotic-arm domains)

Usage:
    export ROBOFLOW_API_KEY=xxxxxxxx
    python download_dataset.py --workspace <ws> --project <proj> --version 1 \
        --format yolov8 --out ./data
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", required=True, help="Roboflow workspace slug")
    parser.add_argument("--project", required=True, help="Roboflow project slug")
    parser.add_argument("--version", type=int, default=1, help="Dataset version number")
    parser.add_argument("--format", default="yolov8", help="Export format (yolov8, yolov5, coco, ...)")
    parser.add_argument("--out", default="./data", help="Output directory")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit(
            "ROBOFLOW_API_KEY is not set. Get a free key from your Roboflow "
            "account settings and `export ROBOFLOW_API_KEY=...` before running this."
        )

    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("Missing dependency: pip install roboflow")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download(args.format, location=args.out)

    print(f"Downloaded to: {dataset.location}")
    print(f"data.yaml expected at: {os.path.join(dataset.location, 'data.yaml')}")


if __name__ == "__main__":
    main()
