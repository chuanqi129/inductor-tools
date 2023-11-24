#!/bin/bash

ins_name=${1:-icx-guilty-search}
ins_id=${2:-0}
ins_type=${3:-c6i.16xlarge}
store_size=${4:-200}
profile_name=${5:-pytorch}

max_ins_num=1
# TODO: use instance config file to instead of hard code
if [ $ins_name == "icx-guilty-search" ]; then
    max_ins_num=20
    ins_type="c6i.16xlarge"
    store_size=200
elif [ $ins_name == "icx-ondemand-tmp" ]; then
    max_ins_num=3
    ins_type="c6i.16xlarge"
    store_size=300
elif [ $ins_name == "icx-regular" ]; then
    max_ins_num=2
    ins_type="c6i.16xlarge"
    store_size=200
elif [ $ins_name == "icx-regular-cppwrapper" ]; then
    max_ins_num=2
    ins_type="c6i.16xlarge"
    store_size=250
elif [ $ins_name == "spr-guilty-search" ]; then
    max_ins_num=5
    ins_type="m7i.16xlarge"
    store_size=200
elif [ $ins_name == "spr-ondemand-tmp" ]; then
    max_ins_num=2
    ins_type="m7i.16xlarge"
    store_size=200
elif [ $ins_name == "spr-regular" ]; then
    max_ins_num=1
    ins_type="m7i.16xlarge"
    store_size=200
fi

# Specify the instance id for regular tests
if [ "$ins_id" != "0" ]; then
    status=`$aws ec2 describe-instance-status --instance-ids ${ins_id} --out text --profile ${profile_name}`
    if [ $? -eq 0 ]; then
        running=`echo $status | grep running | wc -l`
        if [ $running -eq 1 ]; then
            echo "waiting_instance"
        else
            echo ${ins_id}
        fi
    else
        echo "waiting_instance"
    fi
else
    $aws ec2 describe-instances --query "Reservations[*].Instances[*].{PublicDnsName: PublicDnsName, InstanceType: InstanceType, State: State.Name, Name:Tags[?Key=='Name']|[0].Value, InstanceId:InstanceId}" --filters Name=tag:Name,Values=${ins_name} --output text --profile ${profile_name} > ${WORKSPACE}/${ins_name}.txt

    run_num=`cat ${WORKSPACE}/${ins_name}.txt | grep running | wc -l`

    if [ $run_num -lt $max_ins_num ]; then
        bash create_instance.sh $ins_type $ins_name $store_size
    else
        echo "waiting_instance"	
    fi
fi
