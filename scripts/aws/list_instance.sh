# Please install aws-cli and config sso in adavance
profile_name=${1:-pytorch}

scp -r yudongsi@mlt-ace:~/.aws ~/ > /dev/null
aws ec2 describe-instances --query "Reservations[*].Instances[*].{PublicDnsName: PublicDnsName, InstanceType: InstanceType, State: State.Name, Name:Tags[?Key=='Name']|[0].Value, InstanceId:InstanceId}" --output table --profile $profile_name
