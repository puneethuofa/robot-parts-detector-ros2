"""
Unit tests for the pure-Python pre/post-processing helpers. These don't
need ROS2 or TensorRT installed — run with:

    cd ros2_ws/src/part_detector
    python -m pytest test/test_utils.py -v
"""
import numpy as np
import pytest

from part_detector.utils import letterbox, xywh_to_xyxy, nms, postprocess


def test_letterbox_square_output():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    padded, scale, pad_x, pad_y = letterbox(img, new_shape=640)
    assert padded.shape[:2] == (640, 640)
    assert scale == pytest.approx(640 / 640)


def test_letterbox_preserves_aspect_ratio():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    padded, scale, pad_x, pad_y = letterbox(img, new_shape=640)
    # width is the limiting dimension (640 wide vs 480 tall), so scale = 640/640 = 1.0
    # and vertical padding should be added, horizontal should be ~0
    assert pad_y > 0
    assert pad_x == pytest.approx(0, abs=1)


def test_xywh_to_xyxy():
    boxes = np.array([[10.0, 10.0, 4.0, 6.0]])  # cx, cy, w, h
    xyxy = xywh_to_xyxy(boxes)
    np.testing.assert_allclose(xyxy[0], [8.0, 7.0, 12.0, 13.0])


def test_nms_removes_overlapping_boxes():
    boxes = np.array([
        [0, 0, 10, 10],
        [1, 1, 11, 11],   # heavily overlaps box 0
        [50, 50, 60, 60], # separate box
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.95])
    keep = nms(boxes, scores, iou_thresh=0.5)
    assert 0 in keep
    assert 1 not in keep
    assert 2 in keep
    assert len(keep) == 2


def test_nms_empty_input():
    assert nms(np.zeros((0, 4)), np.zeros((0,))) == []


def test_postprocess_filters_low_confidence():
    # raw_output shape: (1, 4+num_classes, N) -- build 2 candidate boxes,
    # one above and one below the confidence threshold
    num_classes = 2
    n = 2
    raw = np.zeros((1, 4 + num_classes, n), dtype=np.float32)
    # box 0: centered at (320,320), 20x20, class 0 conf 0.9
    raw[0, :, 0] = [320, 320, 20, 20, 0.9, 0.1]
    # box 1: centered at (100,100), 20x20, class 1 conf 0.1 (below thresh)
    raw[0, :, 1] = [100, 100, 20, 20, 0.05, 0.1]

    detections = postprocess(raw, scale=1.0, pad_x=0, pad_y=0, conf_thresh=0.25, num_classes=num_classes)
    assert len(detections) == 1
    x1, y1, x2, y2, score, class_id = detections[0]
    assert class_id == 0
    assert score == pytest.approx(0.9)
    assert x1 == pytest.approx(310.0)
    assert x2 == pytest.approx(330.0)


def test_postprocess_undoes_letterbox_offset():
    raw = np.zeros((1, 4 + 1, 1), dtype=np.float32)
    raw[0, :, 0] = [100, 100, 10, 10, 0.9]  # class 0 only

    # simulate a letterbox that scaled by 0.5 and padded 20px on the x-axis
    detections = postprocess(raw, scale=0.5, pad_x=20, pad_y=0, conf_thresh=0.25, num_classes=1)
    assert len(detections) == 1
    x1, y1, x2, y2, score, class_id = detections[0]
    # (95 - 20) / 0.5 = 150, (105 - 20)/0.5 = 170
    assert x1 == pytest.approx(150.0)
    assert x2 == pytest.approx(170.0)
