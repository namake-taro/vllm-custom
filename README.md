# vLLM custom (for DGX Spark): STREAM LOADING

[日本語](README.ja.md)

* This project is conducted for personal interest, learning, and research purposes. Please use it for research and similar purposes.

## Why STREAM LOADING

* DGX Spark users have faced a difficult situation when new large models are released: they often have to wait a long time for 4-bit pre-quantized versions to become available before they can try them out — even when the model would fit in 128 GiB of memory if only it could be quantized to 4 bits on the fly.
* Because vLLM targets many platforms, it is very hard to extract maximum performance on the DGX Spark memory architecture out of the box.
* For example, when quantizing BF16 weights down to 4 bits, the default vLLM requires both the full BF16 dataset and the converted 4-bit dataset to be held in memory at the same time. Models that don't fit in 128 GiB at the BF16 stage cannot be loaded, even if they would fit comfortably after 4-bit quantization.
* STREAM LOADING removes this "you must read all of BF16 before you can quantize it" constraint by **reading just the necessary expert / layer chunks from storage, quantizing them to 4 bits on the fly, and placing the result on the GPU**, so the model can be loaded using only the memory required by the 4-bit form. (Unfortunately, startup time grows significantly.)


## STREAM LOADING — main feature and supporting features

### STREAM LOADING

* It works not only for models with shards laid out neatly in expert order (such as gpt-oss and Qwen), but also for models like Nemotron whose shards are not in expert order, by using random-access loading to perform stream quantization.
* The load buffer required at startup is only "the size of one quantization unit", which makes it possible to run very large models that would otherwise not fit on a single GB10 node at all.
* Example: **Qwen3.5-397B-A17B-FP8** (weights alone are about 96.7 GiB/GPU at TP=2) runs on DGX Spark.

### NF4 quantization (a sub-mode of MXFP4)

* MXFP4 (E2M1) achieves speedup by reducing data transfer through 4-bit quantization. However, output quality degradation due to quantization error is unavoidable.
* To address this, we implemented **MXFP4(NF4) quantization** as a sub-mode of MXFP4.
* MXFP4 quantizes data into 16 levels (4 bits) **uniformly**. By replacing this uniform spacing with a **16-level partition based on the normal distribution**, we were able to dramatically improve precision.
* It is launched within the `--quantization mxfp4` framework, and you can switch between MXFP4 / NF4 / FP8 per layer using environment variables such as `VLLM_NF4_LAYERS`.

### Automatic KV cache allocation

* Until now, DGX Spark users had to find the optimal `--gpu-memory-utilization` value by trial and error, shifting it by 0.1 at a time, in order to maximize KV cache memory.
* Once STREAM LOADING enables huge models, the weights consume most of the free memory, making this manual tuning even harder.
* So with `auto` (the default), the patch was modified to allocate KV cache size at the maximum of currently available memory.
* It first allocates a minimal KV cache, then after `torch.compile` and FlashInfer JIT compilation, it releases the KV cache, recomputes available memory, and re-allocates KV cache up to the limit. The caching allocator's fragmentation pool is also taken into account, which avoids OOM during inference.
* Users can also configure the margin from the memory limit via an environment variable, to leave headroom for memory used during inference.


## Benchmark results

* You will notice that some of the entries below are configurations that, normally, should be impossible to run at TP=1 or TP=2. (The models below are NOT pre-quantized 4-bit models such as Int4 or NVFP4.)
* These results use default values for most settings, including the layer assignments. Further tuning is likely possible.
* Throughput numbers come from `llama-benchy`.

### Throughput

* Decode throughput, single request (tg128, c1, tokens/s)

| Model | TP=1 | TP=2 |
|--------|------|------|
| gpt-oss-120b | 64.52 | **79.55** |
| Qwen3.5-35B-A3B | 64.37 | **78.45** |
| Qwen3.5-27B | 12.07 | **20.46** |
| Qwen3.5-122B-A10B | 28.17 | **41.90** |
| Nemotron3-120B-A12B-BF16 | 24.11 | **36.58** |
| Qwen3.5-397B-A17B-FP8 | - | **26.83** |

* Decode throughput, 10 concurrent requests (tg128, c10, tokens/s, total)

| Model | TP=1 | TP=2 |
|--------|------|------|
| gpt-oss-120b | 165.02 | 198.19 |
| Qwen3.5-35B-A3B | 208.31 | 161.37 |
| Qwen3.5-27B | 92.54 | 95.18 |
| Qwen3.5-122B-A10B | 74.90 | 75.11 |
| Nemotron3-120B-A12B-BF16 | 84.56 | 61.48 |
| Qwen3.5-397B-A17B-FP8 | - | 50.98 |


### KV cache size

| Model | TP=1 KV cache (tokens) | TP=1 max concurrency | TP=2 KV cache (tokens) | TP=2 max concurrency |
|--------|------------------------|----------------------|------------------------|----------------------|
| gpt-oss-120b | 962,352 | 11.74x | 3,395,392 | 41.41x |
| Qwen3.5-35B-A3B | 1,852,864 | 26.80x | 4,106,064 | 59.38x |
| Qwen3.5-27B | 500,192 | 7.33x | 1,204,224 | 17.67x |
| Qwen3.5-122B-A10B | 538,704 | 7.48x | 2,405,376 | 33.41x |
| Nemotron3-120B-A12B-BF16 | 772,992 | 11.44x | 4,456,320 | 65.99x |
| Qwen3.5-397B-A17B-FP8 | - | - | 91,872 | 2.32x |

* The default leaves a 1 GiB margin for OOM safety.

* `max concurrency` is the multiplier for how many `max_model_len`-token prompts can fit simultaneously.


## Installation

### Prerequisites

- NVIDIA DGX Spark (GB10 / SM121)
- Python 3.12
- CUDA 13.0
- (For TP=2) Two DGX Spark nodes + multi-node communication setup
- vLLM custom (based on vLLM 0.17.1)

### Step 1: Create venv

```bash
python3 -m venv ~/.python-vllm-custom
source ~/.python-vllm-custom/bin/activate
```

### Step 2: Install vLLM custom

```bash
# vLLM 0.17.1-based custom wheel
pip install --extra-index-url https://download.pytorch.org/whl/cu130 \
  https://github.com/namake-taro/vllm-custom/releases/download/v0.17.1+sparkcustom.00efb251a6/vllm-0.17.1+sparkcustom.00efb251a6.precompiled-cp312-cp312-linux_aarch64.whl
pip install 'nvidia-nccl-cu13>=2.29.2' > /dev/null 2>&1
```

> The complete modified source package is available on the [Releases page](https://github.com/namake-taro/vllm-custom/releases) as `.src.tar.gz`.

### Step 3: Clear caches

```bash
rm -rf ~/.cache/flashinfer/
rm -rf ~/.cache/vllm/torch_compile_cache/
rm -rf ~/.cache/vllm/torch_aot_compile/
rm -rf /tmp/torchinductor_$(whoami)/
```

> Note: Always clear the caches before each `vllm serve`. Stale caches can cause infinite-loop generation in long-form output even with otherwise correct configurations.


## How to run and configure

* For most BF16 / FP8 models, as long as the 4-bit-quantized form fits in memory, you can run them just by adding `--quantization mxfp4` to `vllm serve`, with the default settings or only minor changes. You no longer need to set `--gpu-memory-utilization` (the default is now `auto`).

### Environment variables

```bash
# === Required for STREAM LOADING ===

# Enable / disable stream loading (ON=1, OFF=0)
export VLLM_STREAM_LOADING=1

# GB10 does not support GDS (GPUDirect Storage), so set this to 1
export VLLM_FASTSAFETENSORS_NOGDS=1

# MXFP4 backend (marlin is required for correct output)
export VLLM_MXFP4_BACKEND=marlin


# Switch quantization method per layer (comma-separated, multiple entries allowed)
# Priority: FP8 > NF4 > MXFP4
# `all` applies to every quantizable layer
export VLLM_FP8_LAYERS=
export VLLM_NF4_LAYERS=all
export VLLM_MXFP4_LAYERS=

# NF4 quantization group size
# 32, 64, 128, 256, 512 (32 gives the best output quality)
export VLLM_NF4_GROUP_SIZE=128

# Free memory margin (MiB) reserved when allocating KV cache.
# The default leaves headroom for safety.
# However, shrinking the margin too much can cause OOM even during llama-benchy runs.
export VLLM_KV_CACHE_MEM_MARGIN=1024

# Other example settings
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
export NCCL_NET_GDR_LEVEL=5
export NCCL_PROTO=LL,LL128,Simple
export NCCL_SOCKET_IFNAME=enp1s0f1np1
```


## Example launch commands

### openai/gpt-oss-120b (TP=1)

```bash
vllm serve openai/gpt-oss-120b \
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-seqs 10 \
  --max-num-batched-tokens 32768 \
  --max-cudagraph-capture-size 32 \
  --max-model-len 131072 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --reasoning-parser openai_gptoss \
  --kv-cache-dtype fp8 \
  --quantization mxfp4 \
  --load-format fastsafetensors \
  --tensor-parallel-size 1 \
  --swap-space 0
```

### Qwen/Qwen3.5-397B-A17B-FP8 (TP=2)

> **Note:** For extreme cases like 397B, where weights exceed 96 GiB per node, you need to keep `--max-num-batched-tokens` and `--max-cudagraph-capture-size` small to minimize transient warmup memory consumption. For other smaller models, the usual values around `32768` / `32` are fine.

```bash
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-seqs 10 \
  --max-num-batched-tokens 4176 \
  --max-cudagraph-capture-size 8 \
  --language-model-only \
  --kv-cache-dtype fp8 \
  --max-model-len 131072 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --quantization mxfp4 \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --swap-space 0
```


## About per-layer quantization assignments

### Quick command to display layer info

* You can use the `tools/list_layers.py` script bundled in the repository to inspect the quantizable layers of a model.

```
./tools/list_layers.py openai/gpt-oss-120b
```

This produces the layer information for openai/gpt-oss-120b:

```
Layer Name             Category         N Shape                  Size
------------------------------------------------------------------------
embed_tokens           embedding        1 (201088, 2880)         1104.6 MB
k_proj                 linear          36 (512, 2880)            2.8 MB
o_proj                 linear          36 (2880, 4096)           22.5 MB
q_proj                 linear          36 (4096, 2880)           22.5 MB
router                 linear          36 (128, 2880)            720 KB
v_proj                 linear          36 (512, 2880)            2.8 MB
lm_head                lm_head          1 (201088, 2880)         1104.6 MB
down_proj_bias         moe_packed      36 (128, 2880)            720 KB
down_proj_blocks       moe_packed      36 (128, 2880, 90, 16)    506.2 MB
gate_up_proj_bias      moe_packed      36 (128, 5760)            1.4 MB
gate_up_proj_blocks    moe_packed      36 (128, 5760, 90, 16)    1012.5 MB
```

### Example environment variable settings based on layer info

* For example, with gpt-oss-120b you might use the following settings, based on the output above.
* Priority: FP8 > NF4 > MXFP4.

```bash
export VLLM_FP8_LAYERS=o_proj,lm_head
export VLLM_NF4_LAYERS=moe,q_proj,k_proj,v_proj
export VLLM_MXFP4_LAYERS=all
```


## Acknowledgements

* This work is built on top of the efforts of the NVIDIA Developer Forum DGX Spark community, who identified and documented the DGX Spark, GB10, and SM121 issues.
Reference: [Computing / DGX Spark / GB10 User Forum topics - NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/)


## License

* These sources are provided as-is to the DGX Spark community. The vLLM project itself is licensed under Apache-2.0.
