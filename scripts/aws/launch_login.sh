#!/usr/bin/env bash
set -x
FF=${1:-0}
GD=${2:-0}
USER=${3:-0}
PASSWD=${4:-0}

timeout 5s ./enter_code.sh
FF_PID=$(ps -ef | grep firefox | grep -v grep | awk '{print $2}' | awk -F '/' '{print $1}')
SSO_PID=$(ps -ef | grep sso | grep -v grep | awk '{print $2}' | awk -F '/' '{print $1}')
if [ -n "${FF_PID:1}" ]; then
    kill -9 ${FF_PID:1}
    echo kill ${FF_PID:1} successful
else
    echo not running ${FF_PID:1}
fi
if [ -n "${SSO_PID:1}" ]; then
    kill -9 ${SSO_PID:1}
    echo kill ${SSO_PID:1} successful
else
    echo not running ${SSO_PID:1}
fi
enter_code=$(sed -n '8p' aws_sso_entercode.log)
echo $enter_code
python aws_sso.py -f ${FF} -d ${GD} -c $enter_code -u ${USER} -p ${PASSWD}
