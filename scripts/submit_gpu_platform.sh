#! /bin/bash

OPTIONS_NCCL="NCCL_DEBUG=VERSION NCCL_IB_HCA=mlx5_ CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_IB_DISABLE=0 NCCL_IB_TIMEOUT=23 NCCL_IB_RETRY_CNT=7"

# run training scripts
source $1

mkdir -p logs/${MLP_TASK_ID}

run_cmd="${OPTIONS_NCCL} torchrun --nproc_per_node $MLP_GPU --master_addr $MLP_WORKER_0_HOST --node_rank $MLP_ROLE_INDEX --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM ${script_path} ${gpt_options}"
echo $run_cmd
eval ${run_cmd} 2>&1 | tee logs/${MLP_TASK_ID}/${MLP_ROLE_INDEX}.log
