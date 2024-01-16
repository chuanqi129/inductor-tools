ins_id=${1:-123}
ins_name=${2:icx-test}
profile_name=${3:-pytorch}

aws ec2 create-tags --resources ${ins_id} --tags Key=Name,Value=${ins_name} --profile ${profile_name}

echo $ins_id
