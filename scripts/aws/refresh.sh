#!/usr/bin/env bash
set -x
AWS=${1}
FF=${2}
GD=${3}
IAP=${4}
PASSWD=${5}
REMOTE=${6}
WA=${7}

cat /dev/null >nohup.out
nohup echo -e '\n\n\n\n\n\n\n\n' | $AWS configure sso --profile pytorch &>nohup.out &
sleep 10s
FF_PID=$(ps -ef | grep firefox/firefox | grep $USER | grep -v refresh | grep -v grep | awk '{print $2}' | awk -F '/' '{print $1}')
# SSO_PID=$(ps -ef | grep sso | grep -v grep | awk '{print $2}' | awk -F '/' '{print $1}')
if [ -n "${FF_PID}" ]; then
    kill -9 ${FF_PID}
    echo kill ${FF_PID} successful
else
    echo not running ${FF_PID}
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
    if [ "${REMOTE}" == "False" ] && [ "${WA}" == "True" ]; then
        sshpass -p${PASSWD} scp aws_sso.py chuanqiw@aws-sso-remote:~/
        sshpass -p${PASSWD} ssh chuanqiw@aws-sso-remote "bash -c 'python aws_sso.py -f /localdisk/chuanqiw/firefox/firefox -d /localdisk/chuanqiw/geckodriver -c $enter_code -u ${IAP} -p ${PASSWD} > ~/aws_sso_refresh.log' &"
        sleep 30s
        sshpass -p${PASSWD} ssh chuanqiw@aws-sso-remote "cat ~/aws_sso_refresh.log"
    else
        python aws_sso.py -f ${FF} -d ${GD} -c $enter_code -u ${IAP} -p ${PASSWD}
    fi
else
    echo Not Found enter_code, SSO configure may be still valid
fi
