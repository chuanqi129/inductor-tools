#!/usr/bin/bash
set -x
conda install -y -c conda-forge gcc_linux-64=13.1.0 gxx_linux-64=13.1.0
ln -sf /opt/conda/libexec/gcc/x86_64-conda-linux-gnu/13.1.0/gcc /usr/bin/gcc
which gcc
gcc --version