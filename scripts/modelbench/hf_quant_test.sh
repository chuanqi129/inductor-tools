if [ ! -d "./logs" ]; then
  mkdir -p "./logs"
fi

model_all=${1:-text}
sw_stack=${2:-torch_compile_quant} # the sw stack you use, this controls what additional options to add
numa_mode=${3:-throughput}
additional_options=$4

if [ ${model_all} == "all" ]; then
    model_all="\
    text-classification+bert-base-cased,\
    text-classification+prajjwal1/bert-tiny,\
    text-classification+prajjwal1-bert-mini,\
    text-classification+bert-large-cased,\
    text-classification+distilbert-base-cased,\
    text-classification+albert-base-v1,\
    text-classification+roberta-base,\
    text-classification+xlnet-base-cased,\
    text-classification+xlm-roberta-base,\
    text-classification+google/electra-base-generator,\
    text-classification+google/electra-base-discriminator,\
    text-classification+allenai/longformer-base-4096,\
    text-classification+google/mobilebert-uncased,\
    text-classification+bert-base-chinese,\
    text-classification+distilbert-base-uncased-finetuned-sst-2-english,\
    text-classification+mrm8488/bert-tiny-finetuned-sms-spam-detection,\
    text-classification+microsoft/MiniLM-L12-H384-uncased,\
    token-classification+bert-base-cased,\
    token-classification+distilbert-base-cased,\
    token-classification+albert-base-v1,\
    token-classification+roberta-base,\
    token-classification+xlnet-base-cased,\
    token-classification+xlm-roberta-base,\
    token-classification+google/electra-base-generator,\
    token-classification+google/electra-base-discriminator,\
    multiple-choice+bert-base-cased,\
    multiple-choice+distilbert-base-cased,\
    multiple-choice+albert-base-v1,\
    multiple-choice+roberta-base,\
    multiple-choice+xlnet-base-cased,\
    multiple-choice+xlm-roberta-base,\
    multiple-choice+google/electra-base-generator,\
    multiple-choice+google/electra-base-discriminator,\
    question-answering+bert-base-cased,\
    question-answering+distilbert-base-cased,\
    question-answering+albert-base-v1,\
    question-answering+roberta-base,\
    question-answering+xlnet-base-cased,\
    question-answering+xlm-roberta-base,\
    question-answering+google/electra-base-generator,\
    question-answering+google/electra-base-discriminator,\
    masked-language-modeling+bert-base-cased,\
    masked-language-modeling+distilbert-base-cased,\
    masked-language-modeling+albert-base-v1,\
    masked-language-modeling+roberta-base,\
    masked-language-modeling+xlm-roberta-base,\
    masked-language-modeling+google/electra-base-generator,\
    masked-language-modeling+google/electra-base-discriminator,\
    casual-language-modeling+gpt2,\
    casual-language-modeling+bert-base-cased,\
    casual-language-modeling+roberta-base,\
    casual-language-modeling+xlnet-base-cased,\
    casual-language-modeling+xlm-roberta-base,\
    summarization+t5-small,\
    summarization+t5-base,\
    "
elif [ ${model_all} == "text" ]; then
    model_all="\
    text-classification+bert-base-cased,\
    text-classification+prajjwal1/bert-tiny,\
    text-classification+bert-large-cased,\
    text-classification+distilbert-base-cased,\
    text-classification+albert-base-v1,\
    text-classification+roberta-base,\
    text-classification+xlnet-base-cased,\
    text-classification+xlm-roberta-base,\
    text-classification+google/electra-base-generator,\
    text-classification+google/electra-base-discriminator,\
    text-classification+google/mobilebert-uncased,\
    text-classification+bert-base-chinese,\
    text-classification+distilbert-base-uncased-finetuned-sst-2-english,\
    text-classification+mrm8488/bert-tiny-finetuned-sms-spam-detection,\
    text-classification+microsoft/MiniLM-L12-H384-uncased,\
    "
elif  [ ${model_all} == "key" ]; then
    model_all="\
    text-classification+albert-base-v1,\
    "
fi

model_list=($(echo "${model_all}" |sed 's/,/ /g'))

# this script handles no-jit situation (most huggingface cannot jit)
if [ ${sw_stack} == "pt" ]; then
    additional_options="${additional_options} --ipex True --channels_last 1 "
elif [ ${sw_stack} == "torch_compile" ]; then
    additional_options="${additional_options} --torch_compile --report_to=none "
elif [ ${sw_stack} == "torch_compile_quant" ]; then
    additional_options="${additional_options} --torch_compile --torch_compile_quant ptq_dynamic --report_to=none "
elif [ ${sw_stack} == "torch_compile_quant_static" ]; then
    additional_options="${additional_options} --torch_compile --torch_compile_quant ptq --report_to=none "
fi

export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

sockets_num=$(lscpu |grep 'Socket(s):' |sed 's/[^0-9]//g')
cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
phsical_cores_num=$( echo "${sockets_num} * ${cores_per_socket}" |bc )
numa_nodes_num=$(lscpu |grep 'NUMA node(s):' |sed 's/[^0-9]//g')
cores_per_node=$( echo "${phsical_cores_num} / ${numa_nodes_num}" |bc )
cpu_model="$(lscpu |grep 'Model name:' |sed 's/.*: *//')"

if [ ${numa_mode} == "throughput" ]; then
    ncpi=${cores_per_node}
    num_instances=1
    batch_size=$(echo "${cores_per_node} * 2" | bc)
elif [ ${numa_mode} == "latency" ]; then
    ncpi=1
    num_instances=${cores_per_node}
    batch_size=1
elif [ ${numa_mode} == "multi_instance" ]; then
    ncpi=4
    num_instances=$(echo "${cores_per_node} / ${ncpi}" | bc)
    batch_size=1
fi
additional_options="${additional_options} --per_device_eval_batch_size ${batch_size} "
numa_launch_header=" python -m numa_launcher --ninstances ${num_instances} --ncore_per_instance ${ncpi} "

for model in ${model_list[@]}
do
    IFS="+"
    array=(${model})
    task_name=${array[0]}
    model_name=${array[1]}
    IFS=$' \t\n'

    echo ">>>>>>>---------------------------${model}------------------------<<<<<<<"

    if [ ${task_name} == "text-classification" ]; then
        workload_launch_cmd="./transformers/examples/pytorch/${task_name}/run_glue.py \
                              --model_name_or_path ${model_name} \
                              --task_name MRPC \
                              --do_eval \
                              --max_seq_length 16 \
                              --learning_rate 2e-5 \
                              --overwrite_output_dir \
                              --output_dir /tmp/tmp_huggingface/ "
    elif [ ${task_name} == "token-classification" ]; then
        workload_launch_cmd="./transformers/examples/pytorch/${task_name}/run_ner.py \
                              --model_name_or_path ${model_name} \
                              --dataset_name conll2003 \
                              --do_eval \
                              --overwrite_output_dir \
                              --output_dir /tmp/tmp_huggingface/ "
    elif [ ${task_name} == "multiple-choice" ]; then
        workload_launch_cmd="./transformers/examples/pytorch/${task_name}/run_swag.py \
                              --model_name_or_path ${model_name} \
                              --do_eval \
                              --learning_rate 5e-5 \
                              --overwrite_output_dir \
                              --output_dir /tmp/tmp_huggingface/ "
    elif [ ${task_name} == "question-answering" ]; then
        workload_launch_cmd="./transformers/examples/pytorch/${task_name}/run_qa.py \
                              --model_name_or_path ${model_name} \
                              --dataset_name squad \
                              --do_eval \
                              --max_seq_length 384 \
                              --learning_rate 3e-5 \
                              --doc_stride 128 \
                              --overwrite_output_dir \
      --output_dir /tmp/tmp_huggingface/ "
    elif [ ${task_name} == "masked-language-modeling" ]; then
        workload_launch_cmd="./transformers/examples/pytorch/language-modeling/run_mlm.py \
                              --model_name_or_path ${model_name} \
                              --dataset_name wikitext \
                              --dataset_config_name wikitext-2-raw-v1 \
                              --do_eval \
                              --overwrite_output_dir \
                              --output_dir /tmp/tmp_huggingface/ "
    elif [ ${task_name} == "casual-language-modeling" ]; then
        workload_launch_cmd="./transformers/examples/pytorch/language-modeling/run_clm.py \
                              --model_name_or_path ${model_name} \
                              --dataset_name wikitext \
                              --dataset_config_name wikitext-2-raw-v1 \
                              --do_eval \
                              --overwrite_output_dir \
                              --output_dir /tmp/tmp_huggingface/ "
    elif [ ${task_name} == "summarization" ]; then
        workload_launch_cmd="./transformers/examples/pytorch/${task_name}/run_summarization.py \
                              --model_name_or_path ${model_name} \
                              --do_eval \
                              --dataset_name xsum \
                              --source_prefix 'summarize: ' \
                              --output_dir /tmp/tmp_huggingface/ \
                              --overwrite_output_dir \
                              --predict_with_generate "
    fi
	
    # handles model names like xxx/yyy to make it xxx-yyy for log filename sake
    if [[ ${model} =~ "/" ]]; then
        model=${model//'/'/'-'}
    fi

    ${numa_launch_header} ${workload_launch_cmd} ${additional_options} 2>&1 | tee ./logs/${model}-${numa_mode}-${sw_stack}.log
    #throughput=$(grep "Throughput:" ./logs/${model}-${precision}-${numa_mode}-${sw_stack}.log | sed -e 's/.*Throughput//;s/[^0-9.]//g' | awk 'BEGIN {sum = 0;}{sum = sum + $1;} END {printf("%.3f", sum);}')
    #echo broad_huggingface ${model} ${precision} ${numa_mode} ${sw_stack} ${throughput} | tee -a ./logs/summary.log
done

