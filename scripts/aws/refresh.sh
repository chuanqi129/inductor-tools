#!/usr/bin/env bash
set -x
FF=${1:-0}
GD=${2:-0}
USER=${3:-0}
PASSWD=${4:-0}

cat /dev/null >nohup.out
nohup echo -e '\n\n\n\n\n\n\n\n' | $aws configure sso --profile pytorch &>nohup.out &
sleep 8s
FF_PID=$(ps -ef | grep firefox | grep -v grep | awk '{print $2}' | awk -F '/' '{print $1}')
# SSO_PID=$(ps -ef | grep sso | grep -v grep | awk '{print $2}' | awk -F '/' '{print $1}')
if [ -n "${FF_PID:1}" ]; then
    kill -9 ${FF_PID:1}
    echo kill ${FF_PID:1} successful
else
    echo not running ${FF_PID:1}
fi
# if [ -n "${SSO_PID:1}" ]; then
#     kill -9 ${SSO_PID:1}
#     echo kill ${SSO_PID:1} successful
# else
#     echo not running ${SSO_PID:1}
# fi

enter_code=$(grep -A 2 'enter the code' nohup.out | awk 'NR==3')
echo $enter_code
if [ -n "${enter_code}" ]; then
    python aws_sso.py -f ${FF} -d ${GD} -c $enter_code -u ${USER} -p ${PASSWD}
else
    echo Not Found enter_code, SSO configure may be still valid
fi
