PASSWD=${1}

# Create user jenkins
sudo useradd --uid 1000 jenkins
echo jenkins:${PASSWD} |sudo chpasswd
# sudo without password
sudo echo 'jenkins ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/jenkins
# docker permission
sudo usermod -a -G docker jenkins
# xpu device
sudo usermod -a -G render jenkins
 
# Install AWS CLI
rm -rf ./aws-cli && mkdir ./aws-cli && cd ./aws-cli
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
cd ../ && rm -rf ./aws-cli

