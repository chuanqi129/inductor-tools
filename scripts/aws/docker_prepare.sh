#!/bin/bash
set -xe

# install java for jenkins launch
sudo apt install -y openjdk-17-jre-headless

# docker setup
sudo apt update -y
sudo apt install docker.io docker-buildx -y && sudo usermod -aG docker ${USER}
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
