# syntax=docker/dockerfile:1
ARG BASE_IMAGE=ubuntu:20.04
FROM ${BASE_IMAGE} AS dev-base
RUN apt-get update && \
        DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        ffmpeg \
        git \
        libgoogle-perftools-dev \
        libjpeg-dev \
        libpng-dev \
        libsm6 \
        libxext6 \
        numactl \
        pybind11-dev \
        vim \
    && rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM dev-base AS conda
ARG PYTHON_VERSION=3.8
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    # Fix conda install issue
    sed -i "s;#!/bin/sh;#!/bin/bash;" ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda config --set channel_priority strict && \
    /opt/conda/bin/conda config --append channels conda-forge && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} \
        astunparse \
        cffi \
        cython \
        cmake>=3.13.0 \
        dataclasses \
        future \
        git-lfs \
        intel-openmp \
        ipython \
        mkl \
        mkl-include \
        ninja \
        numpy \
        requests \
        typing \
        typing_extensions \
        Pillow \
        pkg-config \
        pybind11 \
        pyyaml \
        setuptools \
        jemalloc \
        openssl \
        weasyprint && \
    /opt/conda/bin/conda clean -ya

FROM dev-base AS build
ARG PT_REPO=https://github.com/blzheng/pytorch
ARG PT_BRANCH=beilei/nightly_nov_9
ARG PT_COMMIT=${PT_BRANCH}
COPY --from=conda /opt/conda /opt/conda
ENV LD_PRELOAD /opt/conda/lib/libiomp5.so:/opt/conda/lib/libjemalloc.so
ENV MALLOC_CONF oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1
WORKDIR /workspace
RUN git clone -b ${PT_BRANCH} ${PT_REPO} && \
    cd pytorch && git checkout ${PT_COMMIT} && git submodule sync && git submodule update --init --recursive && \
    python setup.py develop && cd .. 
ARG TORCH_VISION_BRANCH=nightly
ARG TORCH_VISION_COMMIT=${TORCH_VISION_BRANCH}
ARG TORCH_DATA_BRANCH=nightly
ARG TORCH_DATA_COMMIT=${TORCH_DATA_BRANCH}
ARG TORCH_TEXT_BRANCH=nightly
ARG TORCH_TEXT_COMMIT=${TORCH_TEXT_BRANCH}
ARG TORCH_AUDIO_BRANCH=nightly
ARG TORCH_AUDIO_COMMIT=${TORCH_AUDIO_BRANCH}
RUN git clone -b ${TORCH_VISION_BRANCH} https://github.com/pytorch/vision.git && \
    cd vision && git checkout ${TORCH_VISION_COMMIT} && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && cd .. && \
    git clone -b ${TORCH_DATA_BRANCH} https://github.com/pytorch/data.git && \
    cd data && git checkout ${TORCH_DATA_COMMIT} && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && cd .. && \
    git clone -b ${TORCH_TEXT_BRANCH} https://github.com/pytorch/text.git && \
    cd text && git checkout ${TORCH_TEXT_COMMIT} && git submodule sync && git submodule update --init --recursive && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && cd .. && \
    git clone -b ${TORCH_AUDIO_BRANCH} https://github.com/pytorch/audio.git && \
    cd audio && git checkout ${TORCH_AUDIO_COMMIT} && git submodule sync && git submodule update --init --recursive && \
    # Workaround for https://github.com/pytorch/audio/issues/2784
    sed -i "3 i link_directories(/opt/conda/lib)" CMakeLists.txt && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && cd ..
ARG TORCH_BENCH_BRANCH=main
ARG TORCH_BENCH_COMMIT=${TORCH_BENCH_BRANCH} 
RUN git clone -b ${TORCH_BENCH_BRANCH} https://github.com/pytorch/benchmark.git && \
    cd benchmark && git checkout ${TORCH_BENCH_COMMIT} && pip install --no-deps -r requirements.txt && \
    pip install --no-cache Jinja2==3.1.2 markupsafe==2.0.1 && \
    python install.py && \
    # Need update numpy version to avoid some models crash
    pip install --upgrade numpy
ARG BENCH_COMMIT=${PT_COMMIT}
WORKDIR /workspace/pytorch
RUN git checkout ${BENCH_COMMIT} benchmarks && \
    python -c "import torch, torchvision, torchtext"

# Clean for final image
FROM dev-base AS image
ARG BENCH_COMMIT=${PT_COMMIT}
COPY --from=build /opt/conda /opt/conda
COPY --from=build /workspace/pytorch /workspace/pytorch
COPY --from=build /workspace/benchmark /workspace/benchmark
WORKDIR /workspace/pytorch
