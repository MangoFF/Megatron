#!/bin/bash

VOCAB_FILE=/workspace/mango/Megatron/data/gpt2-vocab.json
MERGE_FILE=/workspace/mango/Megatron/data/gpt2-merges.txt
DATA_PATH=/workspace/mango/Megatron/data/CodeData-gpt2_text_document

EXP_NAME="13b"

MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=32

TP_SIZE=2
PP_SIZE=1

NHIDDEN=5120
FFN_HIDDEN=20480
NLAYERS=40
NHEADS=40
SEQ_LEN=2048

SAVE_INTERVAL=2500

TRAIN_TOKENS=10000000000
TRAIN_SAMPLES=$((TRAIN_TOKENS / SEQ_LEN))
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES * 90 / 100))
LR_WARMUP_SAMPLES=$((TRAIN_SAMPLES * 1 / 100))

script_path="pretrain_gpt.py"

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
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --ffn-hidden-size $FFN_HIDDEN \
    --seq-length $SEQ_LEN \
    --num-attention-heads $NHEADS \
    --max-position-embeddings $SEQ_LEN \
    --disable-bias-linear \
    --use-flash-attn \
"
#    --transformer-impl transformer_engine \
#    --fp8-hybrid \
#    --swiglu \
#    --no-position-embedding \
#    --use-rotary-position-embeddings \
#    --untie-embeddings-and-output-weights \

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
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --eval-iters 10 \
    --eval-interval 1000 \
    --save-interval $SAVE_INTERVAL \
    --timing-log-level 2 \
    --log-timers-to-tensorboard \
    --log-memory-to-tensorboard \
"

gpt_options="
    $MODEL_ARGS \
    $TRAINING_ARGS \
    $OPTIMIZER_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-timeout-minutes 60 \
    --init-method-std 0.01 \
"
