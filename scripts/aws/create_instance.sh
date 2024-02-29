#/bin/bash

ins_type=${1:-c6i.16xlarge}
ins_name=${2:icx-test}
store_size=${3:-250}
profile_name=${4:-pytorch}
image_id=${5:-ami-0fc5d935ebf8bc3bc}

ins=`$aws ec2 run-instances \
    --image-id ${image_id} \
    --count 1 \
    --instance-type ${ins_type} \
    --key-name icx-key \
    --security-group-ids sg-08bde89f3858b8350 \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${store_size},\"DeleteOnTermination\":true}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=auto-create-instance}]" \
    --profile ${profile_name}`

ins_id=`echo ${ins} | grep -Po 'InstanceId": "i-[0-9,a-z]+(?=",)' | awk -F ": \"" '{print $2}'`

$aws ec2 create-tags --resources ${ins_id} --tags Key=Name,Value=${ins_name} --profile ${profile_name}

echo $ins_id

