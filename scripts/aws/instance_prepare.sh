set +e

# docker setup
sudo apt update -y
sudo snap install docker
if [[ ! $(groups "${USER}" | grep -q docker) ]]; then
  sudo groupadd docker
  sudo gpasswd -a ${USER} docker
  sudo su
  exit
  newgrp docker <<TEST
echo "Running docker as ${USER} in docker group to see if it's working..."
docker run hello-world
TEST
else
  echo "docker group already exists and contains ${USER}"
fi

# inductor weights prepare
if [ -f /home/ubuntu/docker/download/hub/checkpoints/inductor_weights_prepared.txt ]; then
    echo "Inductor weights prepared"
else
    mkdir -p /home/ubuntu/docker/download/hub/checkpoints
    cp /home/ubuntu/inductor_weights.sh /home/ubuntu/docker/download/hub/checkpoints/
    cd /home/ubuntu/docker/download/hub/checkpoints/
    bash inductor_weights.sh
fi
