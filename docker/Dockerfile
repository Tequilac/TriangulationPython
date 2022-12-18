FROM ubuntu:22.04 as base

RUN apt-get update &&\
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        libopencv-dev \
        libgtest-dev \
        cmake \
        make \
        g++ \
        libgtk2.0-dev \
        pkg-config \
        wget \
        unzip \
    && \
    apt-get install --reinstall -y ca-certificates

WORKDIR /root

RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.6.0.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.6.0.zip && \
    unzip opencv_contrib.zip && \
    unzip opencv.zip && \
    mkdir -p build && cd build && \
    cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.6.0/modules ../opencv-4.6.0 && \
    cmake --build . && \
    make install


RUN git clone https://github.com/PavelNajman/Triangulation.git && \
    cd Triangulation && \
    echo "include_directories(/usr/local/include/opencv4)" >> CMakeLists.txt && \
    mkdir -p build && \
    cd build && \
    cmake -S .. -B . && \
    make install

FROM ubuntu:22.04

COPY --from=base /usr/local /usr/local
COPY --from=base /root/Triangulation/build/ /root/Triangulation/build

CMD ["/root/Triangulation/build/Triangulation"]
