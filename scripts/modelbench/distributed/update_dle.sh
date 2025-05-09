set -x

tee > /tmp/oneAPI.repo << EOF
[oneAPI]
name=Intel® oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
EOF

mv /etc/yum.repos.d/oneAPI.repo /etc/yum.repos.d/oneAPI.repo.back
mv /tmp/oneAPI.repo /etc/yum.repos.d
yum install intel-deep-learning-essentials -y
