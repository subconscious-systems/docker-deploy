# Docker deployments

Recipes for serving models with Subconscious SGLang from prebuilt Docker images. Each
subfolder targets one GPU type and holds its own guide (`README.md`) and launch script
(`launch.sh`).

| Folder | GPUs | Model | Default image |
|---|---|---|---|
| [`h100/`](h100/README.md) | H100 / H200 (`sm_90`) | Qwen3.6-27B, Subconscious x-mode | `subconsciouslabs/orangeline-dell:sm_90-v0.2` |

## Quick start

```bash
cd h100
./launch.sh                      # uses every GPU on the host
DETACH=1 ./launch.sh             # same, in the background (logs: docker logs -f subc)
GPUS=0,1 PORT=8001 NAME=subc-2 ./launch.sh # uses GPUs 0 and 1, serves on port 8001, container named subc-2
```

The launch script finds the GPUs, picks the tensor-parallel size, sizes the memory pools,
and starts an OpenAI-compatible server on port 8000 (or the port specified with `PORT`). Once the log shows the server is ready:

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3.6-27B", "messages": [{"role": "user", "content": "Hello"}]}'
```

Settings such as `GPUS`, `PORT`, `MODEL` and `ENABLE_HICACHE` are passed as environment
variables; the folder's README lists them all.

## Host requirements

All deployments need:

- **NVIDIA driver and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).**
  If `docker run` fails with `could not select device driver "" with capabilities: [[gpu]]`,
  the toolkit is missing or not registered with Docker:
  `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.
- **Access to the `subconsciouslabs` images on Docker Hub.** The repositories are private;
  run `docker login` with the credentials we share before the first launch.
- **Model weights**, either in a Hugging Face cache on the host or in a local directory.
  Each folder's README gives the details.

## Images

Images are built from the SubconsciousCache-SGLang repository and tagged
`<sm_arch>-v<release>`, for example `sm_90-v0.1`. An image only runs on the GPU architecture
in its tag: an `sm_90` image will not start on B200 (`sm_100`) or A100 (`sm_80`). Use the
folder that matches your GPUs.
