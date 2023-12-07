#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

export NCCL_IB_DISABLE=0 
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1   
export NCCL_SOCKET_IFNAME=bond0 
export NCCL_DEBUG=None

GPUS_PER_NODE=8
NNODES=1

# Change for multinode config
HOST_FILE_PATH="../hostfile"
MASTER_ADDR=localhost
MASTER_PORT=6000

WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
EXP_NAME="13b"

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=32

TP_SIZE=4
PP_SIZE=1

NHIDDEN=8192
FFN_HIDDEN=22016
NLAYERS=10
NHEADS=64
SEQ_LEN=2048

TRAIN_TOKENS=10000000000
TRAIN_SAMPLES=$((TRAIN_TOKENS / SEQ_LEN))
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES * 90 / 100))
LR_WARMUP_SAMPLES=$((TRAIN_SAMPLES * 1 / 100))

SAVE_INTERVAL=2500

CHECKPOINT_PATH=$HOME/TE/Megatron-LM/output
VOCAB_FILE=$HOME/TE/Megatron-LM/data/gpt2-vocab.json
MERGE_FILE=$HOME/TE/Megatron-LM/data/gpt2-merges.txt
DATA_PATH=$HOME/TE/Megatron-LM/data/CodeData-gpt2_text_document

DISTRIBUTED_ARGS="
    -np $WORLD_SIZE \
    --hostfile ${HOST_FILE_PATH} \
    -bind-to none -map-by slot  \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
"

OPTIMIZER_ARGS="
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 1.5e-4 \
    --min-lr 1.5e-5 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    --hidden-dropout 0.0 \
    --attention-dropout 0.0 \
    --initial-loss-scale 65536 \
"

MODEL_ARGS="
    --bf16 \
    --transformer-impl transformer_engine \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --ffn-hidden-size $FFN_HIDDEN \
    --seq-length $SEQ_LEN \
    --num-attention-heads $NHEADS \
    --max-position-embeddings $SEQ_LEN \
    --disable-bias-linear \
"

TRAINING_ARGS="
    --tp-comm-overlap \
    --use-flash-attn \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --use-distributed-optimizer \
    --sequence-parallel \
    --recompute-activations \
    --recompute-granularity selective \
    --no-gradient-accumulation-fusion \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-throughput \
    --log-interval 1 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --save-interval 1000 \
    --log-timers-to-tensorboard \
    --log-memory-to-tensorboard \
"

mpirun $DISTRIBUTED_ARGS python pretrain_gpt.py \
    $MODEL_ARGS \
    $TRAINING_ARGS \
    $OPTIMIZER_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-timeout-minutes 60 \
    --init-method-std 0.01 \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH

