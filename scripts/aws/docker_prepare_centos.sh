set +e

# docker setup
sudo yum update -y
sudo yum install -y java-11-openjdk-headless # java for jenkins
# install docker guild: https://docs.docker.com/engine/install/centos/#installation-methods
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker ${USER}
if [[ ! $(groups "${USER}" | grep -q docker) ]]; then
  sudo groupadd docker
  sudo gpasswd -a ${USER} docker
  sudo su
  exit
  newgrp docker <<TEST
TEST
else
  echo "docker group already exists and contains ${USER}"
fi

sudo systemctl start docker
sudo systemctl enable docker