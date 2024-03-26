ins_id=${1}
act=${2:start} # start / stop / terminate
profile_name=${3:-pytorch}

if [[ $act == "start" ]]; then
  aws ec2 start-instances --instance-ids $ins_id --profile $profile_name
elif [[ $act == "stop" ]]; then
  aws ec2 stop-instances --instance-ids $ins_id --profile $profile_name
elif [[ $act == "terminate" ]]; then
  aws ec2 terminate-instances --instance-ids $ins_id --profile $profile_name
else
  echo "Please check act type, make sure in {start, stop, terminate}"
fi

sleep 30s

aws ec2 describe-instances --query "Reservations[*].Instances[*].{PublicDnsName: PublicDnsName, InstanceType: InstanceType, State: State.Name, Name:Tags[?Key=='Name']|[0].Value, InstanceId:InstanceId}" --filters Name=instance-id,Values="$ins_id" --output table --profile $profile_name
