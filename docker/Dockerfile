# syntax=docker/dockerfile:1
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE} AS dev-base
RUN apt-get update && \
        DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        bc \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        ffmpeg \
        git \
        libtcmalloc-minimal4 \
        libjpeg-dev \
        libpng-dev \
        libsm6 \
        libxext6 \
        numactl \
        pybind11-dev \
        wget \
        vim \
    && rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM dev-base AS conda
ARG PYTHON_VERSION=3.10
RUN curl -fsSL -v -k -o ~/miniforge.sh -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    chmod +x ~/miniforge.sh && \
    # Fix conda install issue
    ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} \
        astunparse \
        cffi \
        cython \
        cmake=3.27 \
        mamba=1.5.8 \
        dataclasses \
        future \
        git-lfs \
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
    /opt/conda/bin/conda clean -ya && \
    # Fix GLIBCXX version issue
    rm -f /opt/conda/lib/libstdc++.so.6

FROM dev-base AS build
ARG PT_REPO=https://github.com/pytorch/pytorch
ARG PT_COMMIT=main
COPY --from=conda /opt/conda /opt/conda
WORKDIR /workspace
RUN pip install intel-openmp mpmath==1.3.0
RUN git clone ${PT_REPO} && \
    cd pytorch && git checkout ${PT_COMMIT} && git submodule sync && git submodule update --init --recursive && \
    python setup.py develop && cd ..
ARG TORCH_VISION_COMMIT=default
ARG TORCH_DATA_COMMIT=default
ARG TORCH_TEXT_COMMIT=default
ARG TORCH_AUDIO_COMMIT=default
RUN git clone https://github.com/pytorch/vision.git && \
    cd vision && if [ "${TORCH_VISION_COMMIT}" = "default" ]; then git checkout `cat /workspace/pytorch/.github/ci_commit_pins/vision.txt`; else git checkout ${TORCH_VISION_COMMIT}; fi && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && cd .. && \
    git clone https://github.com/pytorch/data.git && \
    cd data && if [ "${TORCH_DATA_COMMIT}" = "default" ]; then git checkout `cat /workspace/pytorch/.github/ci_commit_pins/data.txt`; else git checkout ${TORCH_DATA_COMMIT}; fi && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && cd .. && \
    git clone https://github.com/pytorch/text.git && \
    cd text && if [ "${TORCH_TEXT_COMMIT}" = "default" ]; then git checkout `cat /workspace/pytorch/.github/ci_commit_pins/text.txt`; else git checkout ${TORCH_TEXT_COMMIT}; fi && git submodule sync && git submodule update --init --recursive && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && cd .. && \
    git clone https://github.com/pytorch/audio.git && \
    cd audio && if [ "${TORCH_AUDIO_COMMIT}" = "default" ]; then git checkout `cat /workspace/pytorch/.github/ci_commit_pins/audio.txt`; else git checkout ${TORCH_AUDIO_COMMIT}; fi && git submodule sync && git submodule update --init --recursive && \
    # Workaround for https://github.com/pytorch/audio/issues/2784
    sed -i "3 i link_directories(/opt/conda/lib)" CMakeLists.txt && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && cd ..
ARG TORCH_BENCH_COMMIT=default
ARG TRANSFORMERS_VERSION=default
ARG HF_HUB_TOKEN=hf_xxx
ENV HUGGING_FACE_HUB_TOKEN ${HF_HUB_TOKEN}
RUN git clone https://github.com/zxd1997066/benchmark.git && \
    cd benchmark && if [ "${TORCH_BENCH_COMMIT}" = "default" ]; then git checkout `cat /workspace/pytorch/.github/ci_commit_pins/torchbench.txt`; else git checkout ${TORCH_BENCH_COMMIT}; fi && pip install --no-deps -r requirements.txt && \
    pip install --no-cache Jinja2==3.1.2 markupsafe==2.0.1 beartype==0.15.0 mpmath==1.3.0 botocore && \
    if [ -f 'torchbenchmark/models/sam_fast/requirements.txt' ]; then sed -i '/github/d' torchbenchmark/models/sam_fast/requirements.txt; fi && \
    python install.py --continue_on_fail && cd .. && \
    echo ${TRANSFORMERS_VERSION} && if [ "${TRANSFORMERS_VERSION}" = "default" ]; then \
        export TRANSFORMERS_COMMIT=`cat /workspace/pytorch/.ci/docker/ci_commit_pins/huggingface.txt`; \
        pip install --force-reinstall git+https://github.com/huggingface/transformers@${TRANSFORMERS_COMMIT}; \
    else \
        pip install --force-reinstall transformers==${TRANSFORMERS_VERSION}; \
    fi && \
    # Need update numpy version to avoid some models crash
    # numpy 2.0 ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
    pip install numpy==1.26.4
ARG BENCH_COMMIT=${PT_COMMIT}
WORKDIR /workspace/pytorch
RUN git checkout ${BENCH_COMMIT} benchmarks && \
    python -c "import torch, torchvision, torchaudio"

# Clean for final image
FROM dev-base AS image
ARG BENCH_COMMIT=${PT_COMMIT}
COPY --from=build /opt/conda /opt/conda
COPY --from=build /workspace/pytorch /workspace/pytorch
COPY --from=build /workspace/vision /workspace/vision
COPY --from=build /workspace/data /workspace/data
COPY --from=build /workspace/text /workspace/text
COPY --from=build /workspace/audio /workspace/audio
COPY --from=build /workspace/benchmark /workspace/benchmark
WORKDIR /workspace/pytorch
