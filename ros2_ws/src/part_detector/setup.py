from setuptools import find_packages, setup

package_name = "part_detector"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/detector.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="Real-time industrial parts detector: TensorRT-optimized YOLO detector wrapped as a ROS2 node.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "detector_node = part_detector.detector_node:main",
        ],
    },
)
