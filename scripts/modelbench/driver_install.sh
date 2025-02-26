set -x
# llama 3-1/1024

#
#PROMPT="write me a story about a boy and his bear"
#
#
#
apt update
apt install -y gpg-agent wget

. /etc/os-release
if [[ ! " jammy " =~ " ${VERSION_CODENAME} " ]]; then
    echo "Ubuntu version ${VERSION_CODENAME} not supported"
else
    wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu ${VERSION_CODENAME}/lts/2350 unified" | \
    tee /etc/apt/sources.list.d/intel-gpu-${VERSION_CODENAME}.list
    apt update
fi
# . /etc/os-release
# if [[ ! " jammy " =~ " ${VERSION_CODENAME} " ]]; then
#     echo "Ubuntu version ${VERSION_CODENAME} not supported"
# else
#     wget https://repositories.intel.com/gpu/ubuntu/dists/jammy/lts/2350/intel-gpu-ubuntu-${VERSION_CODENAME}-2350.run
#     chmod +x intel-gpu-ubuntu-${VERSION_CODENAME}-2350.run
#     ./intel-gpu-ubuntu-${VERSION_CODENAME}-2350.run -y
# fi
apt install -y flex bison intel-fw-gpu intel-i915-dkms xpu-smi
    # linux-headers-$(uname -r) \
    # linux-modules-extra-$(uname -r) \
    #flex bison \
    #intel-fw-gpu intel-i915-dkms xpu-smi
apt install -y intel-i915-dkms || true
apt install -y \
    intel-opencl-icd intel-level-zero-gpu level-zero \
    intel-media-va-driver-non-free libmfxgen1 libvpl2 \
    libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
    libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
    mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo hwinfo clinfo
apt install -y \
    libigc-dev intel-igc-cm libigdfcl-dev libigfxcmrt-dev level-zero-dev
