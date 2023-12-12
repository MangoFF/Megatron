#!/bin/bash
# This example will start serving the 345M model.
DISTRIBUTED_ARGS="--nproc_per_node 4 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT=/share/home/wg/TE/Megatron-LM/output
VOCAB_FILE=/share/home/wg/TE/Megatron-LM/data/gpt2-vocab.json
MERGE_FILE=/share/home/wg/TE/Megatron-LM/data/gpt2-merges.txt
export CUDA_DEVICE_MAX_CONNECTIONS=1

pip install flask-restful

torchrun $DISTRIBUTED_ARGS tools/run_text_generation_server.py   \
       --transformer-impl transformer_engine \
       --tensor-model-parallel-size 4  \
       --pipeline-model-parallel-size 1  \
       --num-layers 5  \
       --hidden-size 8192  \
       --ffn-hidden-size 22016 \
       --disable-bias-linear \
       --use-flash-attn \
       --load ${CHECKPOINT}  \
       --num-attention-heads 64  \
       --max-position-embeddings 2048  \
       --seq-length 2048 \
       --tokenizer-type GPT2BPETokenizer  \
       --bf16   \
       --micro-batch-size 1  \
       --vocab-file $VOCAB_FILE  \
       --merge-file $MERGE_FILE  \
       --seed 42
