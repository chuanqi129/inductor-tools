#!/usr/bin/bash
set -x
conda install -y -c conda-forge gcc_linux-64=13.3 gxx_linux-64=13.3  gcc=13.3 gxx=13.3
# hardcode for the libstdc++.so.6 library.
ln -sf /opt/conda/lib/libstdc++.so.6.0.33 /lib/x86_64-linux-gnu/libstdc++.so.6