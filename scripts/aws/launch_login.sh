#!/usr/bin/env bash
set -x
FF=${1:-0}
GD=${2:-0}
USER=${3:-0}
PASSWD=${4:-0}

timeout 5s ./enter_code.sh
pkill -f ${FF}
enter_code=$(sed -n '8p' aws_sso_entercode.log)
echo $enter_code
python aws_sso.py -f ${FF} -d ${GD} -c $enter_code -u ${USER} -p ${PASSWD}
