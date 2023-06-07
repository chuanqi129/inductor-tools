set -x
# set timeout for aws sso login just for entercode
if [ -f aws_sso_entercode.log ]; then
    rm aws_sso_entercode.log
fi
touch aws_sso_entercode.log
$aws sso login --profile pytorch 2>&1 | tee aws_sso_entercode.log
