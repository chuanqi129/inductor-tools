import argparse

# classification of the reasons why the vectorization has been disabled.
classification = {
        'to_dtype: dtype torch.int64': 0,
        'to_dtype: dtype torch.int32': 0,
        'to_dtype: dtype torch.float64': 0,
        'to_dtype: dtype torch.uint8' : 0,
        'index_expr': 0,
        'reduction: dtype': 0,
        'torch.bool not loaded as mask': 0,
        'torch.int32 not supported by load': 0,
        'torch.int32 not supported by store': 0,
        'torch.int64 not supported by load': 0,
        'torch.int64 not supported by store': 0,
        'torch.float64 not supported by load': 0,
        'torch.float64 not supported by store': 0,
        'torch.bool not supported by store': 0,
        'constant dtype' : 0,
        'not a loop' : 0,
        'op: truediv': 0,
        'op: bitwise_and' : 0,
        'op: remainder' : 0,
        'op: load_seed' : 0,
        'op: randn' : 0,
        'constant store index' : 0,  
        'store mode: atomic_add': 0,     
        'unknow' : 0
    }

def count_kenels_and_vectorized_kernels(file_path, target_string, kenel_log_start, kenel_log__end):
    count_kernel = 0
    count_vectorized_kernel = 0
    inside_context = False
    flag_find_target_string = False

    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            extract_and_classify_disabled_vectorization(line, i)
            if kenel_log_start in line:
                inside_context = True
                count_kernel += 1
                flag_find_target_string = False
            elif kenel_log__end in line:
                if flag_find_target_string:
                    count_vectorized_kernel += 1
                # else:
                #     print(i)
                inside_context = False
                flag_find_target_string

            if inside_context:
                if target_string in line:
                    flag_find_target_string = True

    return count_vectorized_kernel, count_kernel


def extract_and_classify_disabled_vectorization(line, index):
    if 'Disabled vectorization:' in line:
        if 'to_dtype: dtype torch.int32' in line:
            classification['to_dtype: dtype torch.int32'] += 1
        elif 'to_dtype: dtype torch.int64' in line:
            classification['to_dtype: dtype torch.int64'] += 1
        elif 'to_dtype: dtype torch.uint8' in line:
            classification['to_dtype: dtype torch.uint8'] += 1
        elif 'index_expr' in line:
            classification['index_expr'] += 1
        elif 'op: truediv' in line:
            classification['op: truediv'] += 1
        elif 'reduction: dtype' in line:
            classification['reduction: dtype'] += 1
        elif 'torch.bool not loaded as mask' in line:
            classification['torch.bool not loaded as mask'] += 1
        elif 'torch.int32 not supported by load' in line:
            classification['torch.int32 not supported by load'] += 1
        elif 'torch.int32 not supported by store' in line:
            classification['torch.int32 not supported by store'] += 1
        elif 'torch.int64 not supported by load' in line:
            classification['torch.int64 not supported by load'] += 1
        elif 'torch.int64 not supported by store' in line:
            classification['torch.int64 not supported by store'] += 1
        elif 'torch.bool not supported by store' in line:
            classification['torch.bool not supported by store'] += 1
        elif 'torch.float64 not supported by load' in line:
            classification['torch.float64 not supported by load'] += 1
        elif 'torch.float64 not supported by store' in line:
            classification['torch.float64 not supported by store'] += 1
        elif 'constant dtype' in line:
            classification['constant dtype'] += 1
        elif 'not a loop' in line:
            classification['not a loop'] += 1
        elif 'constant store index' in line:
            classification['constant store index'] += 1
        elif 'to_dtype: dtype torch.float64' in line:
            classification['to_dtype: dtype torch.float64'] += 1
        elif 'store mode: atomic_add' in line:
            classification['store mode: atomic_add'] += 1
        elif 'torch.float64 not supported by load' in line:
            classification['torch.float64 not supported by load'] += 1
        elif 'op: bitwise_and' in line:
            classification['op: bitwise_and'] += 1
        elif 'op: remainder' in line:
            classification['op: remainder'] += 1
        elif 'op: load_seed' in line:
            classification['op: load_seed'] += 1
        elif 'op: randn' in line:
            classification['op: randn'] += 1
        else:
            print(index)
            print(line)
            classification['unknow'] += 1
            

def print_disabled_info_classification():
    for key in classification:
        print(f'Disabled vectorization: {key}: {classification[key]} occurrences')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="log information extract job")
    parser.add_argument("--input_log_file", "--input_log_file", required=True, help="input log file path, e.g. /path/test.log")
    
    args = parser.parse_args()
    log_path = args.input_log_file

    # Replace extracted information
    vectorized_message = 'at::vec::Vectorized'
    kernel_start = 'extern "C" void kernel'
    kernel_end = "''')"

    vectorized_kernel_count, total_kernel_count = count_kenels_and_vectorized_kernels(log_path, vectorized_message, kernel_start, kernel_end)
    vertorize_to_total = vectorized_kernel_count/total_kernel_count
    print(f'The total kernel num is: {total_kernel_count}')
    print(f'The vectorized kernel num is: {vectorized_kernel_count}')
    print(f'The vectorized kernel to total kernels is: {vertorize_to_total}')
    print("\n")
    print('Disabled Vectorization Reasons Classification:')
    print_disabled_info_classification()
    