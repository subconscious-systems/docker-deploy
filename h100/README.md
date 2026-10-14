# Serving Qwen3.6-27B on H100 with the `orangeline` image

This guide starts an SGLang server for **Qwen3.6-27B** with Subconscious x-mode, using the
`subconsciouslabs/orangeline` image. The launch script in this folder, `launch.sh`, finds the GPUs on the host
and picks the tensor-parallel (TP) size and memory settings for you.

## Prerequisites

- **Hardware.** An H100 or H200 host. The image is built for `sm_90` only, so it will not run on
  B200 (`sm_100`) or A100 (`sm_80`). The script checks this before starting.
- **Software.** An NVIDIA driver plus the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  so that `docker run --gpus` works.
- **Registry access.** If the `subconsciouslabs/orangeline` repository is private, and we will share the api key separately.
- **Model weights.** One of:
  - a Hugging Face cache that holds `Qwen/Qwen3.6-27B`, or network access plus `HF_TOKEN` so the
    server can download it (about 55 GB);
  - a local directory with the checkpoint, passed in with `MODEL=/path/to/Qwen3.6-27B`.

## Launch script

[`launch.sh`](launch.sh) in this folder starts the server. It reads its settings from environment
variables (see [Usage](#usage)), detects the GPUs, prints the TP size and memory sizing it chose,
and then runs `docker run`. If you copy it to another host, run `chmod +x launch.sh` there.

TP can only be 1, 2, 4 or 8, because Qwen3.6-27B's attention heads only split evenly at those
sizes. The script picks the largest of these that fits the GPU count: 1 GPU → TP=1,
2–3 GPUs → TP=2, 4–7 GPUs → TP=4, 8 or more → TP=8. **A host with 3 GPUs runs at most TP=2**, on
GPUs 0 and 1; the third GPU stays idle.

## Usage

```bash
./launch.sh                                        # all GPUs on the host
GPUS=0 ./launch.sh                                 # one specific GPU
TP=2 ./launch.sh                                   # first 2 GPUs only
GPUS=4,5 PORT=8001 NAME=subc-2 ./launch.sh         # a second server on GPUs 4 and 5
MODEL=/home/llm/model/Qwen3.6-27B ./launch.sh      # local weights instead of the HF cache
DETACH=1 ./launch.sh && docker logs -f subc        # run in the background

# Turn off HiCache (on by default)
ENABLE_HICACHE=0 ./launch.sh
```

Every setting can be overridden with an environment variable:

| Variable | Default | Meaning |
|---|---|---|
| `GPUS` | all GPUs from `nvidia-smi` | Comma-separated GPU indices to choose from |
| `TP` | largest of 8/4/2/1 that fits | Tensor-parallel size; uses the first `TP` GPUs of `GPUS` |
| `IMAGE` | `subconsciouslabs/orangeline-dell:sm_90-v0.1` | Docker image |
| `MODEL` | `Qwen/Qwen3.6-27B` | HF repo id, or a local directory (mounted read-only) |
| `HF_CACHE` | `$HF_HOME` or `~/.cache/huggingface` | Host HF cache mounted at `/hf` |
| `SERVED_NAME` | directory name of `MODEL` | Model name exposed by the API (local `MODEL` only) |
| `PORT` / `NAME` | `8000` / `subc` | Host port and container name |
| `MEM_FRAC` | per-TP table below | `--mem-fraction-static` |
| `MAX_MAMBA` | per-TP table below | `--max-mamba-cache-size` |
| `ENABLE_HICACHE` | `1` | Host-memory L2 cache; `0` turns it off |
| `HICACHE_SIZE` | auto, from free host memory | Host cache size in GB **per TP rank** |
| `BACKTRACK_MAX_LEVELS` | `1` | `SGLANG_SUBCONSCIOUS_BACKTRACK_MAX_LEVELS` (Baseten uses 2) |
| `ST_BUFFER_SIZE` | `10` | `--subconscious-x-st-buffer-size` |
| `DETACH` | `0` | `1` runs the container in the background |
