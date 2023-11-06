#! /bin/bash

NUM_WORKERS=8
NUM_GPUS_PER_WORKER=8
HOST_FILE_PATH="../hostfile"

TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')

source $1
mkdir -p logs/${EXP_NAME}

mpi_cmd="mpirun -np $(($NUM_WORKERS*$NUM_GPUS_PER_WORKER)) \
        --hostfile ${HOST_FILE_PATH} \
        --allow-run-as-root -bind-to none -map-by slot \
        -x MASTER_ADDR=node1 \
        -x MASTER_PORT=$(shuf -i 2000-65000 -n 1) \
        -x GLOO_SOCKET_IFNAME=enp86s0f0 \
        -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
        -x NCCL_ALGO=NVLSTree \
        -x NCCL_DEBUG=VERSION \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_NET_GDR_LEVEL=4 \
        -x NCCL_PXN_DISABLE=1 \
        python ${script_path} ${gpt_options}"

echo ${mpi_cmd}
eval ${mpi_cmd} 2>&1 | tee logs/${EXP_NAME}/output.log
