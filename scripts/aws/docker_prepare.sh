set +e

# docker setup
sudo apt update -y
sudo snap install docker
if [[ ! $(groups "${USER}" | grep -q docker) ]]; then
  sudo groupadd docker
  sudo gpasswd -a ${USER} docker
  sudo chmod 666 /var/run/docker.sock
  sudo su
  exit
  newgrp docker <<TEST
TEST
else
  echo "docker group already exists and contains ${USER}"
fi