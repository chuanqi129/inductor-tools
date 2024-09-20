export docker_image="ccr-registry.caas.intel.com/pytorch/pt_inductor:dind"
export JENKINS_NUMA_PATH="/home/sdp/jenkins_numa"
docker run \
	-td \
	--privileged \
	--env http_proxy=${http_proxy} \
	--env https_proxy=${https_proxy} \
	--env no_proxy=${no_proxy} \
	--cpuset-cpus=0-119 \
	--cpuset-mems=0-2 \
	-v ${JENKINS_NUMA_PATH}/numa0:/root/workspace \
	--name docker-in-docker-numa0 \
	${docker_image}

docker run \
	-td \
	--privileged \
	--env http_proxy=${http_proxy} \
	--env https_proxy=${https_proxy} \
	--env no_proxy=${no_proxy} \
	--cpuset-cpus=120-239 \
	--cpuset-mems=3-5 \
	-v ${JENKINS_NUMA_PATH}/numa1:/root/workspace \
	--name docker-in-docker-numa1 \
	${docker_image}