# vLLM 0.17.0 / 0.17.1 Patches for DGX Spark (GB10 / SM121)

[日本語版](README.ja.md)

Patches to enable high-performance MXFP4 quantized inference on NVIDIA DGX Spark (GB10, SM121) with vLLM 0.17.0 / 0.17.1.

Vanilla vLLM 0.17.0 / 0.17.1 pip wheels do not fully support SM121 (compute capability 12.1).
These patches fix MXFP4 quantization, add BF16-to-MXFP4 online quantization, apply per-layer optimal precision,
and fix SM121-specific kernel bugs -- enabling up to **2x decode throughput** for models such as
**Qwen3.5-35B-A3B** and **openai/gpt-oss-120b**.

If you have a DGX Spark and a few minutes to spare, give it a try.

### Key improvements

- **BF16 -> MXFP4 online quantization** for MoE experts, QKV, and lm_head (vanilla only supports pre-quantized models)
- **Per-layer optimal precision**: MXFP4 for QKV/lm_head, FP8 Marlin for o_proj (E2M1 causes quality degradation on o_proj)
- **SM121 Marlin MoE 256-thread kernel fix** (shared memory race producing garbage output at TP=1)
- **`modules_to_not_convert` config bug fix** (vanilla ignores this field, causing quality degradation on gpt-oss-120b)
- **GDN Triton kernel fix** for Qwen3.5-35B-A3B

## Quantization Configuration

Each layer uses the lowest precision that maintains output quality.
Decode is bandwidth-bound on GB10 -- lower precision = faster.

The configuration below was selected by exhaustively testing all reasonable precision combinations per layer,
choosing the one that maximizes performance while maintaining output quality.

| Layer | Precision | Kernel | bytes/param | Notes |
|-------|-----------|--------|-------------|-------|
| **MoE experts** (w1, w2, w3) | MXFP4 (E2M1) | Marlin FP4 | 0.5 + scale | Pre-quantized or online quantized |
| **QKV** (q_proj, k_proj, v_proj) | MXFP4 (E2M1) | Marlin FP4 | 0.5 + scale | Softmax normalizes quantization error |
| **o_proj** | FP8 (E4M3) | Marlin FP8 | 1.0 | E2M1 causes long-form repetition loops |
| **lm_head** | MXFP4 (E2M1) | Marlin FP4 | 0.5 + scale | BF16 fallback when `tie_word_embeddings=True` |
| embed_tokens | BF16 | -- | 2.0 | Embedding gather, not a GEMM |
| router | BF16 | -- | 2.0 | Negligible size (~13 MB) |
| layer_norm | BF16 | -- | 2.0 | Negligible size (~0.4 MB) |

The following kernels were evaluated for each layer, and the optimal combination was selected:
Marlin FP4 (MXFP4), Marlin FP8, torch.\_scaled\_mm (cuBLAS FP8), CUTLASS FP4 (mma.sync.block\_scale),
FlashInfer CUTLASS MXFP4 (TMA WarpSpecialized, incompatible with SM121), and BF16 (unquantized baseline).

Since GB10 decode is memory-bandwidth-bound, Marlin (which reads pre-quantized weights directly)
outperformed CUTLASS FP4 (+28% slower due to activation quantization overhead) and cuBLAS FP8 (-8.6% vs Marlin FP8).
MXFP4 o_proj achieved 81.6 tok/s but caused infinite loops in long generation -- FP8 Marlin (80.6 tok/s) maintains correct output at nearly the same speed.

GB10 memory bandwidth is 273 GB/s. With the above quantization configuration, gpt-oss-120b reads approximately 2.9 GB of active weights per token, giving a theoretical bandwidth ceiling of ~93 tok/s (TP=1).

Assuming each decode step reads all active weights once at 100% peak bandwidth:

```
2.94 (GB/token) / 273 (GB/s) = 10.8 (ms/token)
1000 (ms) / 10.8 (ms/token) = 92.6 (token/sec)
```

The gap between the theoretical 93 tok/s and the measured 62 tok/s is primarily due to GEMM kernel efficiency and non-GEMM compute overhead. This suggests that vLLM-level tuning is close to the practical limit.
Significant further speedup would require kernel optimization, more aggressive quantization, or speculative decoding (e.g. EAGLE3) to generate multiple tokens per decode step.

## Benchmark Results

Measured with `vllm bench serve` (input: 1024 tokens, output: 128 tokens, random dataset).
Hardware: DGX Spark (GB10 x2), 128 GB unified memory per node, RoCE RDMA interconnect.

### Qwen3.5-35B-A3B

BF16 checkpoint. Vanilla runs BF16 inference (no MXFP4 support for BF16 models).
Patched applies BF16 -> MXFP4 online quantization via Marlin backend.

#### Single Request (num-prompts=1, warm)

| Configuration | TPOT (ms) | Throughput (tok/s) | TTFT (ms) |
|---|---|---|---|
| Vanilla BF16 TP=1 | 32.93 | 28.41 | 327.34 |
| Vanilla BF16 TP=2 | 21.68 | 39.71 | 469.77 |
| **Patched MXFP4 TP=1** | **15.18** | **60.06** | **203.52** |
| **Patched MXFP4 TP=2** | **12.92** | **70.83** | **166.28** |

#### 10 Concurrent Requests (num-prompts=10)

| Configuration | TPOT (ms) | Throughput (tok/s) | Total tok/s |
|---|---|---|---|
| Vanilla BF16 TP=1 | 88.86 | 76.92 | 692.24 |
| Vanilla BF16 TP=2 | 74.77 | 91.25 | 821.24 |
| **Patched MXFP4 TP=1** | **44.32** | **172.04** | **1548.40** |
| **Patched MXFP4 TP=2** | **35.20** | **144.79** | **1303.07** |

### openai/gpt-oss-120b

MXFP4 pre-quantized checkpoint. Both vanilla and patched use MXFP4 for MoE experts.
Patched additionally quantizes QKV (MXFP4 Marlin), o_proj (FP8 Marlin), and lm_head (MXFP4 Marlin),
and includes the SM121 Marlin MoE thread fix for correct TP=1 output.

> **Note:** Vanilla vLLM 0.17.0 / 0.17.1 TP=1 produces garbage output on SM121 due to
> Marlin MoE 256-thread kernel shared memory race. Patched fixes this.

#### Single Request (num-prompts=1, warm)

| Configuration | TPOT (ms) | Throughput (tok/s) | TTFT (ms) |
|---|---|---|---|
| Vanilla MXFP4 TP=1 | -- | -- (broken on SM121) | -- |
| Vanilla MXFP4 TP=2 | 19.09 | 48.66 | 206.48 |
| **Patched TP=1** | **15.87** | **61.78** | **56.39** |
| **Patched TP=2** | **12.33** | **79.28** | **48.85** |

#### 10 Concurrent Requests (num-prompts=10)

| Configuration | TPOT (ms) | Throughput (tok/s) | Total tok/s |
|---|---|---|---|
| Vanilla MXFP4 TP=2 | 45.79 | 175.26 | 1577.30 |
| **Patched TP=1** | **66.02** | **148.24** | **1334.13** |
| **Patched TP=2** | **44.15** | **181.31** | **1631.76** |

### Summary

#### Latency (Single Request TPOT)

| Model | TP | Vanilla | Patched | Speedup |
|---|---|---|---|---|
| Qwen3.5-35B-A3B | 1 | 32.93 ms | **15.18 ms** | **+117%** |
| Qwen3.5-35B-A3B | 2 | 21.68 ms | **12.92 ms** | **+68%** |
| gpt-oss-120b | 1 | broken | **15.87 ms** | **Fixed** |
| gpt-oss-120b | 2 | 19.09 ms | **12.33 ms** | **+55%** |

#### Throughput (Single Request, Output tok/s)

| Model | TP | Vanilla | Patched | Speedup |
|---|---|---|---|---|
| Qwen3.5-35B-A3B | 1 | 28.41 | **60.06** | **+111%** |
| Qwen3.5-35B-A3B | 2 | 39.71 | **70.83** | **+78%** |
| gpt-oss-120b | 1 | broken | **61.78** | **Fixed** |
| gpt-oss-120b | 2 | 48.66 | **79.28** | **+63%** |

## What the Patches Fix

### vllm_all.patch (9 files)

| # | File | Description |
|---|------|-------------|
| 1 | `vllm/envs.py` | Add `VLLM_MXFP4_BACKEND` env var (`auto`/`marlin`/`cutlass_fp4`/`triton`) |
| 2 | `vllm/.../quantization/mxfp4.py` | **Main change**: BF16->MXFP4 online quantization for MoE, `Mxfp4LinearMethod` for QKV, `Fp8MarlinOProjLinearMethod` for o_proj, `Mxfp4LMHeadMethod` for lm_head, `from_config()` bug fix for `modules_to_not_convert` |
| 3 | `vllm/.../quantization/utils/mxfp4_utils.py` | Add `mxfp4_e2m1_quantize()` function, remove expert_map assertion |
| 4 | `vllm/.../fused_moe/fused_marlin_moe.py` | **SM121 fix**: Force 128-thread config for w2 GEMM when N>=2048 to avoid shared memory race in 256-thread kernel |
| 5 | `vllm/.../fused_moe/layer.py` | Fix `weight_loader` ndim check for BF16 per-expert tensors |
| 6 | `vllm/.../fused_moe/cutlass_moe.py` | Add SM121 support, allow EP>1, support expert_map |
| 7 | `vllm/.../fused_moe/routing_simulator.py` | **New file**: MoE routing simulator utility |
| 8 | `vllm/.../fla/ops/fused_recurrent.py` | Fix GDN (Gated Delta Net) Triton kernel for Qwen3.5 |
| 9 | `vllm/.../models/gpt_oss.py` | Add `_quantize_moe_weight_mxfp4()` helper |

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

pip install vllm --extra-index-url https://wheels.vllm.ai/0.17.1/cu130 \
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

# Apply vLLM patch (required)
patch -p0 < /path/to/patches/vllm_all.patch

# Apply FlashInfer CUTLASS fix (only needed for CUTLASS_FP4 backend)
patch -p0 < /path/to/patches/flashinfer_cutlass_sfb_layout_fix.patch
```

### Step 4: Clear caches

```bash
rm -rf ~/.cache/flashinfer/
rm -rf ~/.cache/vllm/torch_compile_cache/
rm -rf ~/.cache/vllm/torch_aot_compile/
rm -rf /tmp/torchinductor_$(whoami)/
```

> **Note:** Always clear caches before every `vllm serve` launch. Stale caches can cause infinite repetition loops in long-form generation even with a correct configuration.

### Step 5: Multi-node setup (TP=2 only)

Sync the entire venv to the remote node:

```bash
REMOTE=<remote-node-ip>

rsync -a ~/.python-vllm-custom/ $REMOTE:~/.python-vllm-custom/

# Clear caches on remote node too
ssh $REMOTE "rm -rf ~/.cache/flashinfer/ ~/.cache/vllm/torch_compile_cache/ ~/.cache/vllm/torch_aot_compile/ /tmp/torchinductor_\$(whoami)/"
```

> **Note:** Always clear caches before every `vllm serve` launch. Stale caches can cause infinite repetition loops in long-form generation even with a correct configuration.

## Running

### Environment variables

```bash
# MXFP4 backend (marlin is required for correct output)
export VLLM_MXFP4_BACKEND=marlin
export VLLM_FASTSAFETENSORS_NOGDS=1

# Performance tuning
export VLLM_MARLIN_USE_ATOMIC_ADD=1          # Faster Marlin reduce for small N (SM>=90 BF16 atomicAdd)
export CUDA_DEVICE_MAX_CONNECTIONS=1          # Serialize CUDA streams for better NCCL overlap
export PYTORCH_ALLOC_CONF=expandable_segments:True  # Reduce memory fragmentation

# For multi-node (adjust interface names for your setup)
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
export NCCL_NET_GDR_LEVEL=5
export NCCL_PROTO=LL,LL128,Simple
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export VLLM_HOST_IP=$(ip -o -4 addr show enp1s0f1np1 | awk '{print $4}' | cut -d/ -f1)
export RAY_memory_usage_threshold=0.99
export VLLM_RPC_TIMEOUT=1800

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
```

### Launch commands

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
  --quantization mxfp4 \
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

- **o_proj must use FP8, not MXFP4.** E2M1 quantization error accumulates through the residual path across 36 layers, causing infinite repetition loops in long-form generation. The patch handles this automatically.
- **NCCL version check:** `pip install` may downgrade NCCL to 2.28.9. Always verify with `pip show nvidia-nccl-cu13` after installing any package.
- **Cache clearing:** After applying patches or changing TP size, clear torch.compile and FlashInfer caches on **all nodes**.
- **`--gpu-memory-utilization 0.80`:** GB10 unified memory requires conservative allocation. Lower to 0.70 if OOM occurs during CUDAGraph capture.
- **Quality testing:** Short-form tests (256 tokens) cannot detect the o_proj precision issue. Always validate with long-form streaming generation (1000+ tokens, temperature=0) after any quantization change.
- For gpt-oss-120b vocab TIKTOKEN configuration, please refer to other resources.

## Package Versions

| Package | Version |
|---------|---------|
| vllm | 0.17.0+cu130 or 0.17.1+cu130 |
| torch | 2.10.0+cu130 |
| triton | 3.6.0 |
| nvidia-nccl-cu13 | >= 2.29.2 |
| fastsafetensors | 0.2.2 |
| flashinfer-python | 0.6.4 |
| transformers | 5.3.0 |
| huggingface_hub | 1.5.0 |

This work builds on the efforts of the DGX Spark community who identified and documented these SM121 issues.
See: [DGX Spark SM121 Software Support Discussion (NVIDIA Forums)](https://forums.developer.nvidia.com/t/dgx-spark-sm121-software-support-is-severely-lacking-official-roadmap-needed/357663)

## License

These patches are provided as-is for the DGX Spark community. The underlying vLLM project is licensed under Apache-2.0.
