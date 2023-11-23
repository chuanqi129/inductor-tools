# an initial script for guilt commit search for inductor releated tests

set +e

IMAGE_TAG=${1:-2023_10_19_aws}
PRECISION=${2:-float32}
CHANNELS=${3:-first}
SHAPE=${4:-static}
WRAPPER=${5:-default}
REPORT_URL=${6}

# get latest performance regression report
if [[ ${PRECISION}_${SHAPE}_${WRAPPER} == "float32_static_default" ]]; then
    echo "float32_static_default"
    REPORT_URL="https://inteltf-jenk.sh.intel.com/job/inductor_aws_regular_dashboard/lastSuccessfulBuild/artifact/inductor_log/inductor_perf_regression.html"
elif [[ ${PRECISION}_${SHAPE}_${WRAPPER} == "float32_dynamic_default" ]]; then
    echo "float32_dynamic_default"
    REPORT_URL="https://inteltf-jenk.sh.intel.com/job/inductor_aws_regular_ds/lastSuccessfulBuild/artifact/inductor_log/inductor_perf_regression.html"
elif [[ ${PRECISION}_${SHAPE}_${WRAPPER} == "float32_static_cpp" ]]; then
    echo "float32_static_cpp"
    REPORT_URL="https://inteltf-jenk.sh.intel.com/job/inductor_aws_regular_cppwrapper/lastSuccessfulBuild/artifact/inductor_log/inductor_perf_regression.html"
elif [[ ${PRECISION}_${SHAPE}_${WRAPPER} == "float32_dynamic_cpp" ]]; then
    echo "float32_dynamic_cpp"
    REPORT_URL="https://inteltf-jenk.sh.intel.com/job/inductor_aws_regular_cppwrapper_ds/lastSuccessfulBuild/artifact/inductor_log/inductor_perf_regression.html"
elif [[ ${PRECISION}_${SHAPE}_${WRAPPER} == "amp_static_default" ]]; then
    echo "amp_static_default"
    REPORT_URL="https://jenkins-aten-caffe2.sh.intel.com/job/inductor_locally_regular/lastSuccessfulBuild/artifact/inductor_log/inductor_perf_regression.html"
fi

rm -rf ${PRECISION}_${SHAPE}_${WRAPPER}_latest_report.html
wget -O ${PRECISION}_${SHAPE}_${WRAPPER}_latest_report.html $REPORT_URL

# prepare
docker pull ccr-registry.caas.intel.com/pytorch/pt_inductor:$IMAGE_TAG
docker run -tid --name $USER --privileged --env https_proxy=${https_proxy} --env http_proxy=${http_proxy} --net host --shm-size 1G -v /home2/yudongsi/.cache:/root/.cache ccr-registry.caas.intel.com/pytorch/pt_inductor:$IMAGE_TAG
wget -O inductor_single_run.sh https://raw.githubusercontent.com/chuanqi129/inductor-tools/yudong/aws_auto/scripts/modelbench/inductor_single_run.sh
docker cp inductor_single_run.sh $USER:/workspace/pytorch
docker cp ${PRECISION}_${SHAPE}_${WRAPPER}_latest_report.html $USER:/workspace/pytorch
docker cp run_test.sh $USER:/workspace/pytorch
rm -rf  inductor_single_run.sh

echo "==============================Start searching guilty commit in ${PRECISION}_${SHAPE}_${WRAPPER} job===================================="
docker exec -ti $USER bash -c "export TRANSFORMERS_OFFLINE=1; \
                               mkdir -p guilty_commit_logs; \
                               chmod +x run_test.sh; \
                               git reset --hard HEAD && git checkout main && git pull && git checkout 543a763cd8b433fc5740ce2b9db15b98e83ed9c2 && git checkout 0200b11 benchmarks; \
                               git reset --hard HEAD && git bisect start 543a763cd8b433fc5740ce2b9db15b98e83ed9c2 02f6a8126e6453d1f5fba585fa7d552f0018263b; \
                               git bisect run ./run_test.sh 0.008208547 multiple inference performance torchbench basic_gnn_gcn float32 first static default 0 on 2>&1 | tee guilty_commit_logs/multiple_inference_performance_torchbench_basic_gnn_gcn_float32_first_static_default_0_on.log; \
                               git bisect reset;\
                               bad_commit=$(cat guilty_commit_logs/multiple_inference_performance_torchbench_basic_gnn_gcn_float32_first_static_default_0_on.log | grep "bisect found first" | awk -F' ' '{ print $9 }')
                               echo '=============first bad commit for basic_gnn_gcn is $bad_commit================'
                               "