#!/bin/bash

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
#export LOG_LEVEL=${LOG_LEVEL:-INFO}
#export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-19}
#export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
#export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}
#export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-2097152}
#export NCCL_AVOID_RECORD_STREAMS=${NCCL_AVOID_RECORD_STREAMS:-1}

CHECKPOINT_PATH=${1:-"checkpoints/llama3_8b_fp8"}
TENSORBOARD_LOGS_PATH=${2:-"tensorboard_logs/llama3_8b_fp8"}
TOKENIZER_ARG=${3:-"MOCK"} # Path to tokenizer model, or "MOCK"
DATA_ARG=${4:-"MOCK"}     # Data prefix, or "MOCK"

# Create directories if they don't exist
mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"

# Distributed training setup
GPUS_PER_NODE=8
NUM_NODES=1
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Path to the pretrain_gpt.py script, assuming this script is run from the root of the Megatron-LM repository
PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

# Fixed training parameters
TP_SIZE=1     
CP_SIZE=1     
PP_SIZE=1     
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
DTYPE="fp8" #"bf16" in case one wants to compute MFU against peak bf16 flops
SEQ_LENGTH=8192
MAX_POSITION_EMBEDDINGS=8192

# Model shape — two knobs for model size sweeps
# Override HIDDEN_SIZE and/or NUM_LAYERS from the environment
HIDDEN_SIZE=${HIDDEN_SIZE:-4096}
NUM_LAYERS=${NUM_LAYERS:-32}

# Derived architecture params (LLaMA3 conventions)
HEAD_DIM=128
NUM_HEADS=$((HIDDEN_SIZE / HEAD_DIM))
GQA_RATIO=4
NUM_QUERY_GROUPS=$(( NUM_HEADS / GQA_RATIO > 0 ? NUM_HEADS / GQA_RATIO : 1 ))
FFN_HIDDEN_SIZE=$((HIDDEN_SIZE * 7 / 2))   # 3.5× for LLaMA3 SwiGLU
INIT_STD=$(echo "scale=6; 1/sqrt($HIDDEN_SIZE)" | bc -l)

# Scaling law sweep parameters
# Override TOTAL_STEPS and PEAK_LR from the environment to vary per isoflop/param config
TOTAL_STEPS=${TOTAL_STEPS:-10000}
LR_WARMUP_STEPS=2000
PEAK_LR=${PEAK_LR:-0.00015}
MIN_LR=$(PEAK_LR="$PEAK_LR" python3 -c "import os; print(f'{float(os.environ[\"PEAK_LR\"]) * 0.1:.2e}')")  # 0.1 × peak

# Derived sample counts (samples = steps × global_batch_size)
TRAIN_SAMPLES=$((TOTAL_STEPS * GLOBAL_BATCH_SIZE))
LR_WARMUP_SAMPLES=$((LR_WARMUP_STEPS * GLOBAL_BATCH_SIZE))
LR_DECAY_SAMPLES=$((TOTAL_STEPS * GLOBAL_BATCH_SIZE))          # full training span: Megatron counts from step 0

# Data cache path (useful for both mock and real data)
DATA_CACHE_PATH="${HOME}/benchmark_cache_llama3_8b_fp8"
mkdir -p "$DATA_CACHE_PATH"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads $NUM_HEADS
    --group-query-attention
    --num-query-groups $NUM_QUERY_GROUPS
    --kv-channels $HEAD_DIM
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --position-embedding-type rope
    --rotary-base 1000000 
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --normalization RMSNorm
    --init-method-std $INIT_STD
    --attention-backend fused
    --apply-layernorm-1p 
    --untie-embeddings-and-output-weights
    --disable-bias-linear 
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples $TRAIN_SAMPLES
    --lr-decay-samples $LR_DECAY_SAMPLES
    --lr-warmup-samples $LR_WARMUP_SAMPLES
    --lr $PEAK_LR
    --min-lr $MIN_LR
    --decoupled-lr $PEAK_LR
    --decoupled-min-lr $MIN_LR
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --grad-reduce-in-bf16
    --cross-entropy-loss-fusion
    --calculate-per-token-loss 
    --manual-gc 
    --empty-unused-memory-level 1 
    #--exit-duration-in-mins 235 
)

# Conditional arguments based on DTYPE (FP8)
DTYPE_ARGS=()
if [[ "$DTYPE" == "fp8" ]]; then
    DTYPE_ARGS+=(
        "--fp8-format hybrid"
        #"--fp8-recipe blockwise" # speedup for B200, but needs higher CUDA
        "--fp8-amax-history-len 1024"
        "--fp8-amax-compute-algo max"
        "--fp8-param-gather"
    )
fi

# Model parallelism arguments
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --context-parallel-size $CP_SIZE
    # --pipeline-model-parallel-size $PP_SIZE # Not explicitly set in llama script options, assume 1 if not multi-node PP
    --sequence-parallel  # Always enable sequence parallelism with TP_SIZE=2
)

# Distributed Data Parallel (DDP) arguments
# From original script's ddp_args
DDP_ARGS=(
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)
TRAINING_ARGS+=("${DDP_ARGS[@]}")


# Data arguments (conditional for mock vs real data)
DATA_ARGS_LIST=()
if [[ "$TOKENIZER_ARG" == "MOCK" ]] || [[ "$DATA_ARG" == "MOCK" ]] || [[ -z "$TOKENIZER_ARG" ]]; then
    DATA_ARGS_LIST+=(
        "--mock-data"
        "--tokenizer-type NullTokenizer"
        "--vocab-size 128256" 
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--tiktoken-pattern v2" 
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
    )
else
    # Settings for real data
    DATA_ARGS_LIST+=(
        "--data-path $DATA_ARG"
        "--tokenizer-type HuggingFaceTokenizer" 
        "--tokenizer-model $TOKENIZER_ARG"
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
        # Note: --vocab-size might be inferred by HuggingFaceTokenizer or might need to be explicit.
        "--vocab-size 128256"
    )
fi

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 32
    --eval-interval 100
    --save-interval 1000
    --save-retain-interval 5000
    --log-throughput
    --profile
    --profile-step-start 4
    --profile-step-end 6
    #--use-pytorch-profiler # if pytorch profiler trace needed
    --ckpt-format torch_dist 
    --distributed-timeout-minutes 60
    --save "$CHECKPOINT_PATH"
    --load "$CHECKPOINT_PATH" 
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
)

# Ensure pretrain_gpt.py is found
if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    echo "Please ensure you are running this script from the root of the Megatron-LM repository, and pretrain_gpt.py is present."
    exit 1
fi

# Run the training command
torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

set +x