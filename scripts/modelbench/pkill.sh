set +e
# kill unused process
itm_1=$(ps -ef | grep entrance | awk '{print $2}')
itm_2=$(ps -ef | grep launch | awk '{print $2}')
itm_3=$(ps -ef | grep inductor | awk '{print $2}')

if [ -n "${itm_1:0}" ]; then
    sudo kill -9 ${itm_1:0}
    echo kill ${itm_1:0} successful
else
    echo not running ${itm_1:0}
fi

if [ -n "${itm_2:0}" ]; then
    sudo kill -9 ${itm_2:0}
    echo kill ${itm_2:0} successful
else
    echo not running ${itm_2:0}
fi

if [ -n "${itm_3:0}" ]; then
    sudo kill -9 ${itm_3:0}
    echo kill ${itm_3:0} successful
else
    echo not running ${itm_3:0}
fi
