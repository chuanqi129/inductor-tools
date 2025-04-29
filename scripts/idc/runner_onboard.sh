TOKEN=${1}

# Prepare Github Action runner scripts
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
echo "29fc8cf2dab4c195bb147384e7e2c94cfd4d4022c793b346a6175435265aa278  actions-runner-linux-x64-2.311.0.tar.gz" | shasum -a 256 -c
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

cd ..
ngpu=$(xpu-smi discovery | grep -c -E 'Device Name')
for i in $(seq 1 $ngpu); do
    let card_num=$i-1
    cp -r actions-runner  actions-runner-${card_num}
    cd actions-runner-${card_num}
    ./config.sh --unattended --url https://github.com/pytorch --token $TOKEN --name `hostname`_pvc_card_${card_num} --runnergroup linux.idc.xpu.group --labels linux.idc.xpu
    ZE_AFFINITY_MASK=$card_num ./run.sh &
    cd ..
done
rm -rf actions-runner

