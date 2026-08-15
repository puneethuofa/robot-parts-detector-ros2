# ROS2 Humble + CUDA/TensorRT dev environment for the part_detector package.
#
# Build:
#   docker build -t part-detector -f docker/Dockerfile .
# Run (with GPU + camera passthrough):
#   docker run --gpus all -it --rm \
#       -v $(pwd):/workspace \
#       --network host \
#       part-detector

FROM nvcr.io/nvidia/tensorrt:24.05-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

# --- ROS2 Humble ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg2 lsb-release software-properties-common \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ros-base \
        ros-humble-vision-msgs \
        ros-humble-cv-bridge \
        ros-humble-launch-ros \
        python3-colcon-common-extensions \
        python3-rosdep \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init && rosdep update

# --- Python deps (ultralytics for training/export scripts, pycuda for TRT bindings) ---
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
COPY . /workspace

SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]
