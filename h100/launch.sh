#!/usr/bin/env bash
# Launch Qwen3.6-27B (Subconscious x-mode) in the orangeline-dell image.
# Detects the host's GPUs and sizes TP / memory pools for them.
set -euo pipefail

IMAGE="${IMAGE:-subconsciouslabs/orangeline-dell:sm_90-v0.2}"
NAME="${NAME:-subc}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B}"   # HF repo id or a local checkpoint directory
HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
ENABLE_HICACHE="${ENABLE_HICACHE:-1}"   # 1 = add host memory as an L2 cache (HiCache); 0 = off
BACKTRACK_MAX_LEVELS="${BACKTRACK_MAX_LEVELS:-1}"
ST_BUFFER_SIZE="${ST_BUFFER_SIZE:-10}"
DETACH="${DETACH:-0}"

# ---- GPU detection ----------------------------------------------------------
if [[ -n "${GPUS:-}" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "$GPUS"
else
    mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
fi
NUM_GPUS=${#GPU_IDS[@]}
if (( NUM_GPUS == 0 )); then
    echo "error: no GPUs found" >&2
    exit 1
fi

# Qwen3.6-27B (24 q / 4 kv heads, 16 / 48 GDN heads) only shards evenly at TP 1, 2, 4 or 8.
if [[ -z "${TP:-}" ]]; then
    for TP in 8 4 2 1; do
        (( TP <= NUM_GPUS )) && break
    done
fi
case "$TP" in
    1|2|4|8) ;;
    *) echo "error: TP must be 1, 2, 4 or 8 (got $TP)" >&2; exit 1 ;;
esac
if (( TP > NUM_GPUS )); then
    echo "error: TP=$TP but only $NUM_GPUS GPU(s) available" >&2
    exit 1
fi
GPU_IDS=("${GPU_IDS[@]:0:TP}")
DEVICES=$(IFS=,; echo "${GPU_IDS[*]}")

while IFS=, read -r idx name cap total used; do
    if [[ "$cap" != "9.0" ]]; then
        echo "error: GPU $idx ($name) is sm_${cap/./}; this image is built for sm_90 (H100/H200)" >&2
        exit 1
    fi
    if (( total < 80000 )); then
        echo "warning: GPU $idx has only $total MiB; the sizing below assumes an 80 GB H100" >&2
    fi
    if (( used > 2048 )); then
        echo "warning: GPU $idx already has $used MiB in use by another process" >&2
    fi
done < <(nvidia-smi --id="$DEVICES" \
    --query-gpu=index,name,compute_cap,memory.total,memory.used \
    --format=csv,noheader,nounits | sed 's/, /,/g')

# ---- Memory sizing per TP (80 GB H100, bf16 weights) -------------------------
# Each running request holds 5 Mamba state slots (extra_buffer + overlap scheduler),
# so max running requests = MAX_MAMBA / 5.
EXTRA_ARGS=()
case "$TP" in
    1) : "${MEM_FRAC:=0.85}" "${MAX_MAMBA:=40}"
       EXTRA_ARGS=(--chunked-prefill-size 4096 --cuda-graph-max-bs 16) ;;
    2) : "${MEM_FRAC:=0.80}" "${MAX_MAMBA:=384}" ;;   # Baseten production values
    4) : "${MEM_FRAC:=0.80}" "${MAX_MAMBA:=768}" ;;
    8) : "${MEM_FRAC:=0.80}" "${MAX_MAMBA:=1536}" ;;
esac

# ---- HiCache (host-memory L2) -----------------------------------------------
HICACHE_ARGS=()
if [[ "$ENABLE_HICACHE" == "1" ]]; then
    if [[ -z "${HICACHE_SIZE:-}" ]]; then
        # SGLang lets each rank pin at most (MemAvailable - 10 GiB) / ranks; use 60% of that.
        avail_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        HICACHE_SIZE=$(( (avail_kb * 1024 - 10 * 1024**3) * 6 / 10 / TP / 1000**3 ))
    fi
    if (( HICACHE_SIZE < 4 )); then
        echo "warning: less than 4 GB of free host memory per rank; HiCache disabled" >&2
    else
        HICACHE_ARGS=(--enable-hierarchical-cache --hicache-size "$HICACHE_SIZE")
    fi
fi

# ---- Model source -----------------------------------------------------------
if [[ -d "$MODEL" ]]; then
    MODEL_MOUNT=(-v "$(realpath "$MODEL"):/models/qwen:ro")
    MODEL_ARGS=(--model-path /models/qwen --served-model-name "${SERVED_NAME:-$(basename "$MODEL")}")
else
    MODEL_MOUNT=(-v "$HF_CACHE:/hf" -e HF_HOME=/hf -e HF_TOKEN)
    MODEL_ARGS=(--model-path "$MODEL")
fi

if [[ "$DETACH" == "1" ]]; then
    RUN_MODE=(--rm -d)
else
    RUN_MODE=(--rm -it)
fi

echo "GPUs: $DEVICES (found $NUM_GPUS, TP=$TP)"
echo "Image: $IMAGE"
echo "Sizing: --mem-fraction-static $MEM_FRAC, --max-mamba-cache-size $MAX_MAMBA" \
     "(max $(( MAX_MAMBA / 5 )) running requests)"
echo "HiCache: ${HICACHE_ARGS[*]:-off}"

docker run --name "$NAME" "${RUN_MODE[@]}" \
    --gpus "\"device=$DEVICES\"" \
    --ipc=host \
    --ulimit memlock=-1 \
    -p "$PORT:8000" \
    "${MODEL_MOUNT[@]}" \
    -e SGLANG_PP_SPIKE_MODE=on_drop \
    -e SGLANG_SUBCONSCIOUS_TRANSPLANT=1 \
    -e SGLANG_SUBCONSCIOUS_BACKTRACK_MAX_LEVELS="$BACKTRACK_MAX_LEVELS" \
    "$IMAGE" \
    python3 -m sglang.launch_server \
        --host 0.0.0.0 --port 8000 \
        "${MODEL_ARGS[@]}" \
        --tp "$TP" \
        --chat-template /sgl-workspace/sglang/deploy/chat_templates/qwen3-coder-raw.jinja \
        --attention-backend flashinfer \
        --page-size 1 \
        --tool-call-parser qwen3_coder \
        --subconscious-x-mode \
        --context-length 262144 \
        --mem-fraction-static "$MEM_FRAC" \
        --max-mamba-cache-size "$MAX_MAMBA" \
        --subconscious-x-st-buffer-size "$ST_BUFFER_SIZE" \
        --subconscious-x-min-span-length 3 \
        --enable-cache-report \
        --stream-response-default-include-usage \
        "${EXTRA_ARGS[@]}" \
        "${HICACHE_ARGS[@]}"
