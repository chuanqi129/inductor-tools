set +e

# inductor weights prepare
if [ -f /home/ubuntu/docker/download/hub/checkpoints/inductor_weights_prepared.txt ]; then
    echo "Inductor weights prepared"
else
    mkdir -p /home/ubuntu/docker/download/hub/checkpoints
    cp /home/ubuntu/inductor_weights.sh /home/ubuntu/docker/download/hub/checkpoints/
    cd /home/ubuntu/docker/download/hub/checkpoints/
    bash inductor_weights.sh
fi
