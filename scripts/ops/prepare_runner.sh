TOKEN=${1}
VERSION=${2-"2.325.0"}
# Prepare Github Action runner scripts
rm -rf actions-runner
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-${VERSION}.tar.gz -L https://github.com/actions/runner/releases/download/v${VERSION}/actions-runner-linux-x64-${VERSION}.tar.gz
tar xzf ./actions-runner-linux-x64-${VERSION}.tar.gz

cd ..
ngpu=$(xpu-smi discovery | grep -c -E 'Device Name')
for i in $(seq 1 $ngpu); do
    let card_num=$i-1
    if [ ! -d actions-runner-${card_num} ]; then
        cp -r actions-runner  actions-runner-${card_num}
    fi
    cd actions-runner-${card_num}
    if [ -f svc.sh ]; then
        sudo ./svc.sh stop
        sudo ./svc.sh uninstall
    fi
    ./config.sh remove --token $TOKEN
    if [ $? -eq 0 ]; then
        ./config.sh --unattended --url https://github.com/pytorch --token $TOKEN --name `hostname`_pvc_card_${card_num} --runnergroup linux.idc.xpu.group --labels linux.idc.xpu
        echo "ZE_AFFINITY_MASK=$card_num" >> .env
        sudo ./svc.sh install $USER
        sudo ./svc.sh start
    fi
    cd ..
done
rm -rf actions-runner