

FROM althack/ros2:humble-gazebo AS sim

# ** [Optional] Uncomment this section to install additional packages. **
#
ENV DEBIAN_FRONTEND=noninteractive
COPY "files/ros2.list" "/etc/apt/sources.list.d/"
COPY "files/sources.list" "/etc/apt/"
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update \
    && apt-get -y install --no-install-recommends \
    ros-humble-rmw-cyclonedds-cpp \
    python3-pytest \
    libgflags-dev nlohmann-json3-dev \
    ros-humble-image-transport ros-humble-image-publisher \
    udev \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    && pip install rosdepc \
    && rosdepc init \
    pip3 install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
ENV DEBIAN_FRONTEND=dialog
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

ARG WORKSPACE
RUN echo "if [ -f ${WORKSPACE}/install/setup.bash ]; then source ${WORKSPACE}/install/setup.bash; fi" >> /home/ros/.bashrc
RUN git config --global --add safe.directory $WORKSPACE
