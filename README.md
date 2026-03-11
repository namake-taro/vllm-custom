# vLLM 0.17.0 Patches for DGX Spark (GB10 / SM121)

Patches to enable MXFP4 quantized inference on NVIDIA DGX Spark (GB10, SM121) with vLLM 0.17.0.

Vanilla vLLM 0.17.0 pip wheels do not fully support SM121 (compute capability 12.1).
These patches fix MXFP4 quantization, multi-node Ray execution, and other SM121-specific issues,
enabling high-performance inference for models such as **Qwen3.5-35B-A3B** and **openai/gpt-oss-120b**.

If you have a DGX Spark and a few minutes to spare, give it a try.

## Benchmark Results

Measured with `vllm bench serve` (input: 1024 tokens, output: 128 tokens, random dataset).
Hardware: DGX Spark (GB10 x2), 128 GB unified memory per node, RoCE RDMA interconnect.

### Qwen3.5-35B-A3B

BF16 checkpoint. Vanilla runs BF16 inference (no MXFP4 support for BF16 models).
Custom applies BF16 -> MXFP4 online quantization via Marlin backend.

#### Single Request (num-prompts=1)

| Configuration | TPOT (ms) | Throughput (tok/s) | TTFT (ms) |
|---|---|---|---|
| Vanilla BF16 TP=1 | 32.93 | 28.41 | 327.34 |
| Vanilla BF16 TP=2 | 21.68 | 39.71 | 469.77 |
| **Custom MXFP4 TP=1** | **15.34** | **59.60** | **199.68** |
| **Custom MXFP4 TP=2** | **12.53** | **72.72** | **168.27** |

#### 10 Concurrent Requests (num-prompts=10)

| Configuration | TPOT (ms) | Throughput (tok/s) | Total tok/s |
|---|---|---|---|
| Vanilla BF16 TP=1 | 88.86 | 76.92 | 692.24 |
| Vanilla BF16 TP=2 | 74.77 | 91.25 | 821.24 |
| **Custom MXFP4 TP=1** | **43.99** | **121.42** | **1092.74** |
| **Custom MXFP4 TP=2** | **32.08** | **153.87** | **1384.86** |

### openai/gpt-oss-120b

MXFP4 pre-quantized checkpoint. Both Vanilla and Custom use MXFP4 for MoE experts.
Custom additionally quantizes attention (qkv_proj, o_proj) and lm_head to FP4,
and includes SM121 Marlin MoE thread fix for correct TP=1 output.

> **Note:** Vanilla vLLM 0.17.0 TP=1 produces malformed output on SM121 due to
> Marlin MoE 256-thread kernel shared memory race. Custom patches fix this.

#### Single Request (num-prompts=1)

| Configuration | TPOT (ms) | Throughput (tok/s) | TTFT (ms) |
|---|---|---|---|
| Vanilla MXFP4 TP=1 | - | - (broken on SM121) | - |
| Vanilla MXFP4 TP=2 | 19.09 | 48.66 | 206.48 |
| **Custom MXFP4 TP=1** | **14.93** | **57.68** | **322.84** |
| **Custom MXFP4 TP=2** | **12.24** | **72.37** | **213.94** |

#### 10 Concurrent Requests (num-prompts=10)

| Configuration | TPOT (ms) | Throughput (tok/s) | Total tok/s |
|---|---|---|---|
| Vanilla MXFP4 TP=2 | 45.79 | 175.26 | 1577.30 |
| **Custom MXFP4 TP=1** | **57.48** | **138.70** | **1248.28** |
| **Custom MXFP4 TP=2** | **38.82** | **197.77** | **1779.96** |

### Summary

#### Latency (Single Request TPOT)

| Model | TP | Vanilla | Custom | Speedup |
|---|---|---|---|---|
| Qwen3.5-35B-A3B | 1 | 32.93 ms | **15.34 ms** | **+115%** |
| Qwen3.5-35B-A3B | 2 | 21.68 ms | **12.53 ms** | **+73%** |
| gpt-oss-120b | 1 | broken | **14.93 ms** | **Fixed** |
| gpt-oss-120b | 2 | 19.09 ms | **12.24 ms** | **+56%** |

#### Throughput (Single Request, Output tok/s)

| Model | TP | Vanilla | Custom | Speedup |
|---|---|---|---|---|
| Qwen3.5-35B-A3B | 1 | 28.41 | **59.60** | **+110%** |
| Qwen3.5-35B-A3B | 2 | 39.71 | **72.72** | **+83%** |
| gpt-oss-120b | 1 | broken | **57.68** | **Fixed** |
| gpt-oss-120b | 2 | 48.66 | **72.37** | **+49%** |

## What the Patches Fix

### vllm_all.patch

| # | File | Description |
|---|------|-------------|
| 1 | `vllm/envs.py` | Add `VLLM_MXFP4_BACKEND` env var (`auto`/`marlin`/`cutlass_fp4`/`triton`) |
| 2 | `vllm/.../quantization/utils/mxfp4_utils.py` | Add `mxfp4_e2m1_quantize()` function, remove expert_map assertion |
| 3 | `vllm/.../quantization/mxfp4.py` | **Main change**: BF16->MXFP4 online quantization for MoE, `Mxfp4LinearMethod` for attention, `Mxfp4LMHeadMethod` for lm_head, `CUTLASS_FP4` backend enum. **Bug fix**: `from_config()` now reads `modules_to_not_convert` from model config |
| 4 | `vllm/.../fused_moe/layer.py` | Fix `weight_loader` ndim check for BF16 per-expert tensors |
| 5 | `vllm/.../fused_moe/fused_marlin_moe.py` | **SM121 fix**: Force 128-thread config for w2 GEMM when N>=2048 to avoid shared memory race in 256-thread kernel. Relax w2 size assertion (`==` -> `>=`) |
| 6 | `vllm/.../fused_moe/cutlass_moe.py` | Add SM121 support (`is_device_capability_family(120)`), allow EP>1, support expert_map |
| 7 | `vllm/.../fused_moe/routing_simulator.py` | **New file**: routing simulator utility for testing/analysis |
| 8 | `vllm/.../fla/ops/fused_recurrent.py` | Fix GDN (Gated Delta Net) Triton kernel for Qwen3.5 |
| 9 | `vllm/.../models/gpt_oss.py` | Add `_quantize_moe_weight_mxfp4()` helper |

### triton_allocator.patch

Fixes Triton's `NullAllocator` crash in Ray distributed workers. The `ContextVar`-based allocator
does not propagate to Ray worker threads, causing `RuntimeError` when Triton kernels require
runtime memory allocation (e.g., FLA's `solve_tril` during GDN prefill). The patch makes
`NullAllocator.__call__` fall back to `torch.cuda.caching_allocator_alloc` instead of raising.
Required for multi-node (TP>=2) inference with Qwen3.5-35B-A3B.

### flashinfer_cutlass_sfb_layout_fix.patch

Fixes a copy-paste bug in FlashInfer 0.6.4's bundled CUTLASS 4.2.1 headers where `layout_SFB`
initialization incorrectly uses `tile_atom_to_shape_SFA`. Only affects CUTLASS_FP4 backend.

## Installation

### Prerequisites

- NVIDIA DGX Spark (GB10 / SM121)
- Python 3.12
- CUDA 13.0

### Step 1: Create venv and install vLLM

```bash
python3 -m venv ~/.python-vllm-custom
source ~/.python-vllm-custom/bin/activate

pip install vllm --extra-index-url https://wheels.vllm.ai/0.17.0/cu130 \
                 --extra-index-url https://download.pytorch.org/whl/cu130
```

### Step 2: Install / upgrade dependencies

```bash
# Upgrade NCCL (pip bundles 2.28.9 which has CUDAGraph + multi-node deadlock bug)
pip install 'nvidia-nccl-cu13>=2.29.2'

# Pin compatible versions
pip install 'transformers==5.3.0'
pip install 'huggingface_hub==1.5.0'

# Install fastsafetensors for fast model loading
pip install fastsafetensors
```

> **Note:** `huggingface_hub >= 1.6.0` causes Harmony parser errors with gpt-oss models. Pin to 1.5.0.

### Step 3: Apply patches

```bash
SITE=~/.python-vllm-custom/lib/python3.12/site-packages
cd "$SITE"

# Apply vLLM patch
patch -p0 < /path/to/patches/vllm_all.patch

# Apply Triton allocator fix (required for multi-node / TP>=2)
patch -p0 < /path/to/patches/triton_allocator.patch

# Apply FlashInfer CUTLASS fix (only needed for CUTLASS_FP4 backend)
patch -p0 < /path/to/patches/flashinfer_cutlass_sfb_layout_fix.patch
```

### Step 4: Clear caches

```bash
rm -rf ~/.cache/triton/
rm -rf ~/.cache/flashinfer/
rm -rf ~/.cache/vllm/torch_compile_cache/
rm -rf ~/.cache/vllm/torch_aot_compile/
rm -rf /tmp/torchinductor_$(whoami)/
```

### Step 5: Multi-node setup (TP=2 only)

Sync the entire venv and caches to the remote node:

```bash
REMOTE=<remote-node-ip>

rsync -a ~/.python-vllm-custom/ $REMOTE:~/.python-vllm-custom/

# Clear caches on remote node
ssh $REMOTE "rm -rf ~/.cache/flashinfer/ ~/.cache/vllm/torch_compile_cache/ ~/.cache/vllm/torch_aot_compile/ /tmp/torchinductor_\$(whoami)/"
```

### Step 6: Set environment variables

```bash
# MXFP4 backend (marlin is required for correct output)
export VLLM_MXFP4_BACKEND=marlin
export VLLM_FASTSAFETENSORS_NOGDS=1

# For multi-node (adjust interface names for your setup)
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
export NCCL_NET_GDR_LEVEL=5
export NCCL_PROTO=LL
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export VLLM_HOST_IP=$(ip -o -4 addr show enp1s0f1np1 | awk '{print $4}' | cut -d/ -f1)
export RAY_memory_usage_threshold=0.99
export VLLM_RPC_TIMEOUT=1800

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
```

### Step 7: Launch

```bash
# Qwen3.5-35B-A3B (BF16 -> MXFP4 online quantization, TP=2)
vllm serve Qwen/Qwen3.5-35B-A3B \
  --served-model-name local-vllm \
  --host 0.0.0.0 \
  --quantization mxfp4 \
  --load-format fastsafetensors \
  --reasoning-parser qwen3 \
  --distributed-executor-backend ray \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.80 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --language-model-only \
  --max-model-len 262144 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 16 \
  --max-cudagraph-capture-size 32

# openai/gpt-oss-120b (MXFP4 pre-quantized, TP=2)
vllm serve openai/gpt-oss-120b \
  --served-model-name local-vllm \
  --host 0.0.0.0 \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --reasoning-parser openai_gptoss \
  --load-format fastsafetensors \
  --distributed-executor-backend ray \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.80 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --max-model-len 131072 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 16 \
  --max-cudagraph-capture-size 32
```

## Important Notes

- **`VLLM_MXFP4_BACKEND=marlin` is required.** The CUTLASS_FP4 backend combined with Linear/LMHead FP4 patches produces degraded output quality.
- **NCCL version check:** `pip install` may downgrade NCCL to 2.28.9. Always verify with `pip show nvidia-nccl-cu13` after installing any package.
- **Cache clearing:** After applying patches or changing TP size, clear torch.compile and FlashInfer caches on **all nodes**.
- **`--gpu-memory-utilization 0.80`:** GB10 unified memory requires conservative allocation. Lower to 0.70 if OOM occurs during CUDAGraph capture.

## Package Versions

| Package | Version |
|---------|---------|
| vllm | 0.17.0+cu130 |
| torch | 2.10.0+cu130 |
| triton | 3.6.0 |
| nvidia-nccl-cu13 | >= 2.29.2 |
| fastsafetensors | 0.2.2 |
| flashinfer-python | 0.6.4 |
| transformers | 5.3.0 |
| huggingface_hub | 1.5.0 |

## Background

NVIDIA DGX Spark (GB10) uses SM121 (compute capability 12.1), which is not fully supported by
standard PyTorch pip wheels (max compute capability 12.0) or vanilla vLLM. Key issues include:

- MXFP4 Marlin MoE 256-thread kernel shared memory race on SM121 (garbage output at TP=1)
- `Mxfp4Config.from_config()` ignoring `modules_to_not_convert` (attention/lm_head incorrectly FP4-quantized)
- Missing PTX instructions (`cvt.rn.satfinite.e2m1x2.f32`) for E2M1 conversion
- FlashInfer CUTLASS MXFP4 backend requiring TMA/wgmma (unavailable on SM121)
- NCCL 2.28.9 CUDAGraph + multi-node deadlock bug
- Triton allocator ContextVar not propagating to Ray worker threads

This work builds on the efforts of the DGX Spark community who identified and documented these SM121 issues.
See: [DGX Spark SM121 Software Support Discussion (NVIDIA Forums)](https://forums.developer.nvidia.com/t/dgx-spark-sm121-software-support-is-severely-lacking-official-roadmap-needed/357663)

## License

These patches are provided as-is for the DGX Spark community. The underlying vLLM project is licensed under Apache-2.0.
