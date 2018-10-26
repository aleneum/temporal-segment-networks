FROM nvidia/cuda:9.2-cudnn7-devel
ENV LIBRARY_PATH=/usr/local/cuda/lib64

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential python-pip cmake git libopenmpi-dev wget \
        libboost-all-dev libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev \
        checkinstall cmake pkg-config yasm libjpeg-dev libavcodec-dev libavformat-dev libswscale-dev \
        libv4l-dev python-dev libtbb-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev \
        libvorbis-dev libxvidcore-dev x264 libgflags-dev libgoogle-glog-dev liblmdb-dev libzip-dev

RUN pip install --upgrade pip && hash -r pip && \
    pip install scikit-learn scikit-image numpy protobuf youtube_dl pyyaml \
                http://download.pytorch.org/whl/cu92/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl \
                torchvision

RUN wget https://cmake.org/files/v3.12/cmake-3.12.3-Linux-x86_64.sh -P / && sh cmake-3.12.3-Linux-x86_64.sh --skip-license --prefix=/usr

# ADD https://cmake.org/files/v3.12/cmake-3.12.3-Linux-x86_64.sh /

RUN git clone --depth 1 --recurse-submodules https://github.com/aleneum/temporal-segment-networks.git /tsn_caffe
RUN wget https://yjxiong.blob.core.windows.net/tsn-init/bn_inception_rgb_init.caffemodel -P /tsn_caffe/models && \
    wget https://yjxiong.blob.core.windows.net/tsn-init/bn_inception_flow_init.caffemodel

# ADD https://yjxiong.blob.core.windows.net/tsn-init/bn_inception_rgb_init.caffemodel \
#    https://yjxiong.blob.core.windows.net/tsn-init/bn_inception_flow_init.caffemodel \
#    /tsn_caffe/models/

RUN mkdir -p /tsn_caffe/3rd-party && \
        git clone --depth 1 --recursive -b 2.4 https://github.com/opencv/opencv /tsn_caffe/3rd-party/opencv && \
        cd /tsn_caffe/3rd-party/opencv && git apply ../../opencv_cuda9.patch
RUN mkdir /tsn_caffe/3rd-party/opencv/build && cd /tsn_caffe/3rd-party/opencv/build && \
        cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_TBB=ON  -D WITH_V4L=ON  -D WITH_CUDA=ON \
              -D WITH_OPENCL=OFF -D CMAKE_CXX_FLAGS="-DTBB_IMPLEMENT_CPP0X" .. && make -j3 && cp lib/cv2.so ../../../

RUN mkdir /tsn_caffe/lib/dense_flow/build && cd /tsn_caffe/lib/dense_flow/build && \
        cmake -D OpenCV_DIR=/tsn_caffe/3rd-party/opencv/build -D CUDA_USE_STATIC_CUDA_RUNTIME=OFF .. && make -j3

RUN mkdir /tsn_caffe/lib/caffe-action/build && cd /tsn_caffe/lib/caffe-action/build && \
         cmake -D USE_MPI=ON -D OpenCV_DIR=/tsn_caffe/3rd-party/opencv/build -D MPI_CXX_COMPILER="/usr/bin/mpicxx" \
               -D CUDA_USE_STATIC_CUDA_RUNTIME=OFF -D CUDA_ARCH_NAME="Manual" -D CUDA_ARCH_BIN="30 35 50 52 60 61 70"  .. && make -j3 install

RUN git clone --depth 1 --recurse-submodules https://github.com/aleneum/tsn-pytorch.git /tsn_pytorch

RUN wget https://yjxiong.blob.core.windows.net/models/bn_inception-9f5701afb96c8044.pth -P /tsn_pytorch/models && \
    wget https://yjxiong.blob.core.windows.net/models/inceptionv3-cuhk-0e09b300b493bc74c.pth -P /tsn_pytorch/models

# ADD https://yjxiong.blob.core.windows.net/models/bn_inception-9f5701afb96c8044.pth \
#    https://yjxiong.blob.core.windows.net/models/inceptionv3-cuhk-0e09b300b493bc74c.pth \
#    /tsn_pytorch/models/
