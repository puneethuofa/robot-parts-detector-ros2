#!/usr/bin/env python3
"""
ROS2 node: subscribes to a camera image topic, runs a TensorRT-optimized
YOLO-style detector, publishes results as a vision_msgs/Detection2DArray,
and (optionally) an annotated debug image for RViz.
"""
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge

from part_detector.trt_inference import TRTEngine
from part_detector.utils import preprocess, postprocess

DEFAULT_CLASS_NAMES = [
    "bolt", "nut", "gear", "bracket", "washer", "pipe_fitting", "part_generic",
]


class PartDetectorNode(Node):
    def __init__(self):
        super().__init__("part_detector_node")

        self.declare_parameter("engine_path", "")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("detections_topic", "/part_detector/detections")
        self.declare_parameter("debug_image_topic", "/part_detector/debug_image")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf_thresh", 0.25)
        self.declare_parameter("iou_thresh", 0.45)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("class_names", DEFAULT_CLASS_NAMES)

        engine_path = self.get_parameter("engine_path").get_parameter_value().string_value
        if not engine_path:
            raise RuntimeError("Parameter 'engine_path' is required (path to .engine file).")

        self.image_topic = self.get_parameter("image_topic").value
        self.detections_topic = self.get_parameter("detections_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value
        self.imgsz = self.get_parameter("imgsz").value
        self.conf_thresh = self.get_parameter("conf_thresh").value
        self.iou_thresh = self.get_parameter("iou_thresh").value
        self.publish_debug_image = self.get_parameter("publish_debug_image").value
        self.class_names = self.get_parameter("class_names").value

        self.get_logger().info(f"Loading TensorRT engine from: {engine_path}")
        self.engine = TRTEngine(engine_path)
        self.bridge = CvBridge()

        # Best-effort/sensor-data QoS matches typical camera driver publishers
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub = self.create_subscription(Image, self.image_topic, self.image_callback, qos)
        self.det_pub = self.create_publisher(Detection2DArray, self.detections_topic, 10)
        if self.publish_debug_image:
            self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 10)

        self._frame_count = 0
        self._t_last_log = time.time()

        self.get_logger().info(
            f"part_detector_node ready. Subscribed to {self.image_topic}, "
            f"publishing to {self.detections_topic}."
        )

    def image_callback(self, msg: Image):
        t0 = time.perf_counter()

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        chw, scale, pad_x, pad_y = preprocess(cv_image, imgsz=self.imgsz)

        outputs = self.engine.infer(chw)
        raw = next(iter(outputs.values()))  # single-output detection head

        detections = postprocess(
            raw, scale, pad_x, pad_y,
            conf_thresh=self.conf_thresh, iou_thresh=self.iou_thresh,
            num_classes=len(self.class_names),
        )

        self._publish_detections(detections, msg.header)
        if self.publish_debug_image:
            self._publish_debug_image(cv_image, detections, msg.header)

        latency_ms = (time.perf_counter() - t0) * 1000
        self._frame_count += 1
        if time.time() - self._t_last_log >= 5.0:
            fps = self._frame_count / (time.time() - self._t_last_log)
            self.get_logger().info(f"~{fps:.1f} FPS | last frame latency: {latency_ms:.1f} ms")
            self._frame_count = 0
            self._t_last_log = time.time()

    def _publish_detections(self, detections, header):
        msg = Detection2DArray()
        msg.header = header

        for (x1, y1, x2, y2, score, class_id) in detections:
            det = Detection2D()
            det.header = header

            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w, h = (x2 - x1), (y2 - y1)
            det.bbox.center.position.x = cx
            det.bbox.center.position.y = cy
            det.bbox.size_x = w
            det.bbox.size_y = h

            hyp = ObjectHypothesisWithPose()
            class_name = (
                self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
            )
            hyp.hypothesis.class_id = class_name
            hyp.hypothesis.score = score
            det.results.append(hyp)

            msg.detections.append(det)

        self.det_pub.publish(msg)

    def _publish_debug_image(self, cv_image, detections, header):
        annotated = cv_image.copy()
        for (x1, y1, x2, y2, score, class_id) in detections:
            pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(annotated, pt1, pt2, (0, 255, 0), 2)
            label = f"{self.class_names[class_id] if class_id < len(self.class_names) else class_id}:{score:.2f}"
            cv2.putText(annotated, label, (pt1[0], max(pt1[1] - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        debug_msg.header = header
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PartDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
