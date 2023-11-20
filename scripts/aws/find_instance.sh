#!/bin/bash

ins_name=${1:-icx-guilty-search}
ins_type=${2:-c6i.16xlarge}
store_size=${3:-200}
profile_name=${4:-pytorch}

max_ins_num=1
# TODO: use instance config file to instead of hard code
if [ $ins_name == "icx-guilty-search" ]; then
    max_ins_num=5
    ins_type="c6i.16xlarge"
    store_size=200
elif [ $ins_name == "icx-ondemand" ]; then
    max_ins_num=3
    ins_type="c6i.16xlarge"
    store_size=200
elif [ $ins_name == "icx-regular" ]; then
    max_ins_num=2
    ins_type="c6i.16xlarge"
    store_size=200
elif [ $ins_name == "icx-regular-cppwrapper" ]; then
    max_ins_num=2
    ins_type="c6i.16xlarge"
    store_size=200
elif [ $ins_name == "spr-ondemand" ]; then
    max_ins_num=3
    ins_type="m7i.16xlarge"
    store_size=200
elif [ $ins_name == "spr-regular" ]; then
    max_ins_num=1
    ins_type="m7i.16xlarge"
    store_size=200
fi

$aws ec2 describe-instances --query "Reservations[*].Instances[*].{PublicDnsName: PublicDnsName, InstanceType: InstanceType, State: State.Name, Name:Tags[?Key=='Name']|[0].Value, InstanceId:InstanceId}" --filters Name=tag:Name,Values=${ins_name} --output text --profile ${profile_name} > ${WORKSPACE}/${ins_name}.txt

run_num=`cat ${WORKSPACE}/${ins_name}.txt | grep running | wc -l`
stop_num=`cat ${WORKSPACE}/${ins_name}.txt | grep stopped | wc -l`

if [[ $run_num < $max_ins_num ]]; then
    if [[ $stop_num > 0 ]]; then
	cat ${WORKSPACE}/${ins_name}.txt | grep stopped | head -n 1 | awk '{print $1}'
    else
	bash create_instance.sh $ins_type $ins_name $store_size
    fi
else
    echo "waiting_instance"	
fi
