#!/bin/bash

# Runs the "345M" parameter model
export CUDA_DEVICE_MAX_CONNECTIONS=1

export NCCL_IB_DISABLE=0 
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1   
export NCCL_SOCKET_IFNAME=bond0 
export NCCL_DEBUG=None

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

MICRO_BATCH_SIZE=16
GLOBAL_BATCH_SIZE=32

TP_SIZE=4
PP_SIZE=1

NHIDDEN=8192
FFN_HIDDEN=22016
NLAYERS=5
NHEADS=128
SEQ_LEN=2048

TRAIN_TOKENS=100000000
TRAIN_SAMPLES=$((TRAIN_TOKENS / SEQ_LEN))
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES * 90 / 100))
LR_WARMUP_SAMPLES=$((TRAIN_SAMPLES * 1 / 100))

SAVE_INTERVAL=2500

CHECKPOINT_PATH=/share/home/wg/TE/BLD/Megatron/output
VOCAB_FILE=$HOME/TE/BLD/Megatron/data/gpt2-vocab.json
MERGE_FILE=$HOME/TE/BLD/Megatron/data/gpt2-merges.txt
DATA_PATH=/share/home/wg/TE/BLD/megatronlm_dataset_autotokenizer/data-sample/out_story_zh_document

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
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
    --use-flash-attn \
"

TRAINING_ARGS="
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
    --log-interval 5 \
    --eval-interval 1000 \
    --eval-iters 5 \
    --save-interval 1000 \
    --log-timers-to-tensorboard \
    --log-memory-to-tensorboard \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
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

