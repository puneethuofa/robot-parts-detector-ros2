#!/usr/bin/env python3
"""
Train a YOLOv8 detector on an industrial-parts dataset.

Thin wrapper around ultralytics so the same script runs identically on a
laptop GPU or on an HPC SLURM allocation (see slurm_train.sbatch).

Usage:
    python train.py --data ./data/data.yaml --model yolov8s.pt \
        --epochs 150 --imgsz 640 --batch 32 --device 0 \
        --project runs/detect --name parts_v1
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to dataset data.yaml")
    parser.add_argument("--model", default="yolov8s.pt", help="Base weights to fine-tune from")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0", help="'0' for GPU 0, 'cpu' for CPU, '0,1' for multi-GPU")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience (epochs)")
    parser.add_argument("--project", default="runs/detect", help="Output project dir")
    parser.add_argument("--name", default="parts_v1", help="Run name")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt in --name run")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        resume=args.resume,
        # Reasonable augmentation defaults for cluttered bin-picking scenes:
        # more rotation/perspective jitter than the YOLO default since parts
        # in a bin appear at arbitrary orientations.
        degrees=15.0,
        perspective=0.0005,
        mosaic=1.0,
        mixup=0.1,
    )

    metrics = model.val()
    print("Validation results:")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
