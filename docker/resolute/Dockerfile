# syntax=docker/dockerfile:1

ARG ROS_IMAGE=ros:rolling-ros-base-resolute
FROM ${ROS_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG PR_NUMBER=19
ARG PR_HEAD_SHA=unknown
ARG BUILD_JOBS=2

LABEL org.opencontainers.image.source="https://github.com/ros2/rosidl_buffer_backends" \
      org.opencontainers.image.revision="${PR_HEAD_SHA}" \
      org.opencontainers.image.title="rosidl_buffer_backends PR ${PR_NUMBER} test environment"

ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    CTEST_PARALLEL_LEVEL=${BUILD_JOBS} \
    ROS_DISTRO=rolling

RUN rosdep update --rosdistro "${ROS_DISTRO}"

WORKDIR /ws
COPY . /ws/src/rosidl_buffer_backends

# The ROS image already provides the generic build toolchain. Every project
# and test dependency below is discovered from package.xml by rosdep.
RUN apt-get update && \
    source "/opt/ros/${ROS_DISTRO}/setup.bash" && \
    rosdep install \
      --from-paths \
        src/rosidl_buffer_backends \
      --ignore-src \
      --rosdistro "${ROS_DISTRO}" \
      --as-root apt:true \
      -y && \
    rm -rf /var/lib/apt/lists/*

# Ubuntu's CUDA metapackage does not add nvcc to PATH. Declare the compiler
# after dependency installation so CMake's CUDA language check can find it.
ENV CUDACXX=/usr/bin/nvcc

RUN source "/opt/ros/${ROS_DISTRO}/setup.bash" && \
    colcon build \
      --merge-install \
      --packages-up-to \
        cuda_buffer_backend cuda_buffer_py \
        onnxruntime_conversions_cpu onnxruntime_conversions_cuda \
        onnxruntime_conversions_py_cpu onnxruntime_conversions_py_cuda \
      --event-handlers console_direct+ \
      --cmake-args -DBUILD_TESTING=ON

RUN source "/opt/ros/${ROS_DISTRO}/setup.bash" && \
    source /ws/install/setup.bash && \
    colcon --log-base /ws/test_log build \
      --base-paths \
        /ws/src/rosidl_buffer_backends/docker/resolute/debian/installed_tests \
      --build-base /ws/test_build \
      --install-base /ws/test_install \
      --packages-select onnxruntime_conversions_installed_tests \
      --cmake-args \
        -DBACKENDS_SOURCE_ROOT=/ws/src/rosidl_buffer_backends

RUN bash /ws/src/rosidl_buffer_backends/docker/resolute/test.sh

CMD ["bash", "/ws/src/rosidl_buffer_backends/docker/resolute/test.sh"]
