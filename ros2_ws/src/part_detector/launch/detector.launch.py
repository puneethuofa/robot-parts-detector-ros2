from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    engine_path_arg = DeclareLaunchArgument(
        "engine_path", description="Absolute path to the TensorRT .engine file"
    )
    image_topic_arg = DeclareLaunchArgument(
        "image_topic", default_value="/camera/image_raw", description="Input camera topic"
    )
    detections_topic_arg = DeclareLaunchArgument(
        "detections_topic", default_value="/part_detector/detections"
    )
    imgsz_arg = DeclareLaunchArgument("imgsz", default_value="640")
    conf_thresh_arg = DeclareLaunchArgument("conf_thresh", default_value="0.25")
    iou_thresh_arg = DeclareLaunchArgument("iou_thresh", default_value="0.45")
    publish_debug_arg = DeclareLaunchArgument("publish_debug_image", default_value="true")

    detector_node = Node(
        package="part_detector",
        executable="detector_node",
        name="part_detector_node",
        output="screen",
        parameters=[{
            "engine_path": LaunchConfiguration("engine_path"),
            "image_topic": LaunchConfiguration("image_topic"),
            "detections_topic": LaunchConfiguration("detections_topic"),
            "imgsz": LaunchConfiguration("imgsz"),
            "conf_thresh": LaunchConfiguration("conf_thresh"),
            "iou_thresh": LaunchConfiguration("iou_thresh"),
            "publish_debug_image": LaunchConfiguration("publish_debug_image"),
        }],
    )

    return LaunchDescription([
        engine_path_arg,
        image_topic_arg,
        detections_topic_arg,
        imgsz_arg,
        conf_thresh_arg,
        iou_thresh_arg,
        publish_debug_arg,
        detector_node,
    ])
