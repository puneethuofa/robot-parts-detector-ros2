"""Preprocessing and postprocessing helpers shared by the ROS2 node and the benchmark script."""
import cv2
import numpy as np


def letterbox(image: np.ndarray, new_shape=640, color=(114, 114, 114)):
    """Resize + pad image to a square target size while preserving aspect ratio.
    Returns the padded image plus (scale, pad_x, pad_y) needed to map boxes back
    to the original image coordinates."""
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    h, w = image.shape[:2]
    scale = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * scale)), int(round(h * scale)))
    pad_w = new_shape[1] - new_unpad[0]
    pad_h = new_shape[0] - new_unpad[1]
    pad_w /= 2
    pad_h /= 2

    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return padded, scale, left, top


def preprocess(bgr_image: np.ndarray, imgsz=640):
    """BGR HWC uint8 -> normalized RGB CHW float32, plus letterbox metadata."""
    padded, scale, pad_x, pad_y = letterbox(bgr_image, imgsz)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(chw), scale, pad_x, pad_y


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    xyxy = np.empty_like(boxes)
    xyxy[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    xyxy[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    xyxy[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    xyxy[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return xyxy


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh=0.45):
    """Standard greedy NMS. boxes: (N,4) xyxy. Returns kept indices."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        remaining = np.where(iou <= iou_thresh)[0]
        order = order[remaining + 1]

    return keep


def postprocess(raw_output: np.ndarray, scale, pad_x, pad_y, conf_thresh=0.25, iou_thresh=0.45, num_classes=None):
    """
    Decode a YOLOv8-style raw output tensor of shape (1, 4+num_classes, N)
    into a list of (x1, y1, x2, y2, score, class_id) in ORIGINAL image coords.
    """
    pred = raw_output[0]  # (4+num_classes, N)
    pred = pred.transpose(1, 0)  # (N, 4+num_classes)

    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]
    if num_classes is not None:
        class_scores = class_scores[:, :num_classes]

    class_ids = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)

    mask = scores >= conf_thresh
    boxes_xywh, scores, class_ids = boxes_xywh[mask], scores[mask], class_ids[mask]
    if len(boxes_xywh) == 0:
        return []

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    keep = nms(boxes_xyxy, scores, iou_thresh)

    detections = []
    for idx in keep:
        x1, y1, x2, y2 = boxes_xyxy[idx]
        # undo letterbox padding/scale to map back to the original image
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale
        detections.append((float(x1), float(y1), float(x2), float(y2), float(scores[idx]), int(class_ids[idx])))

    return detections
