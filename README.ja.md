# vLLM 0.17.0 / 0.17.1 パッチ — DGX Spark (GB10 / SM121) 向け

[English](README.md)

NVIDIA DGX Spark (GB10, SM121) 上で vLLM 0.17.0 / 0.17.1 を使った高性能 MXFP4 量子化推論を実現するパッチです。

vLLM 0.17.0 / 0.17.1 の pip wheel は SM121 (compute capability 12.1) を十分にサポートしていません。
このパッチは MXFP4 量子化の修正、BF16→MXFP4 オンライン量子化の追加、レイヤー別の最適演算精度設定、
SM121 固有のカーネルバグ修正を行い、**Qwen3.5-35B-A3B** や **openai/gpt-oss-120b** などのモデルで
最大 **2倍の decode スループット** を実現します。

DGX Spark をお持ちでしたら、ぜひお試しください。

### 主な改善点

- **BF16→MXFP4 オンライン量子化**: MoE experts、QKV、lm_head に適用（vanilla は事前量子化モデルのみ対応）
- **レイヤー別最適演算精度**: QKV/lm_head に MXFP4、o_proj に FP8 Marlin（E2M1 は o_proj で品質劣化を引き起こす）
- **SM121 Marlin MoE 256-thread カーネル修正**: 共有メモリ競合により TP=1 でゴミ出力が発生する問題を修正
- **`modules_to_not_convert` 設定バグ修正**: vanilla ではこのフィールドが無視され gpt-oss-120b が品質劣化
- **GDN Triton カーネル修正**: Qwen3.5-35B-A3B 向け

## 量子化構成

各レイヤーに対して、出力品質を維持できる最低演算精度を選択しています。
GB10 の decode は帯域律速のため、低演算精度ほど高速です。

下記構成は、合理的に考えられる全ての演算精度の組み合わせをレイヤーごとに網羅的に検証し、
出力品質を維持しつつ最もパフォーマンスが良い組み合わせを選択しました。

| レイヤー | 演算精度 | カーネル | bytes/param | 備考 |
|---------|------|---------|-------------|------|
| **MoE experts** (w1, w2, w3) | MXFP4 (E2M1) | Marlin FP4 | 0.5 + scale | 事前量子化済みまたはオンライン量子化 |
| **QKV** (q_proj, k_proj, v_proj) | MXFP4 (E2M1) | Marlin FP4 | 0.5 + scale | softmax が量子化誤差を正規化するため問題なし |
| **o_proj** | FP8 (E4M3) | Marlin FP8 | 1.0 | E2M1 は長文で繰り返しループを引き起こす |
| **lm_head** | MXFP4 (E2M1) | Marlin FP4 | 0.5 + scale | `tie_word_embeddings=True` の場合は BF16 にフォールバック |
| embed_tokens | BF16 | -- | 2.0 | embedding gather であり GEMM ではない |
| router | BF16 | -- | 2.0 | サイズが極小 (~13 MB) |
| layer_norm | BF16 | -- | 2.0 | サイズが極小 (~0.4 MB) |


各レイヤーに対して以下のカーネルを検証し、最適な組み合わせを選択しています:
Marlin FP4 (MXFP4)、Marlin FP8、torch.\_scaled\_mm (cuBLAS FP8)、CUTLASS FP4 (mma.sync.block\_scale)、
FlashInfer CUTLASS MXFP4 (TMA WarpSpecialized、SM121 非対応)、BF16 (量子化なしベースライン)。

GB10 の decode は帯域律速のため、事前量子化済み重みを直接読む Marlin が、
CUTLASS FP4（activation 量子化オーバーヘッドで +28% 遅い）や cuBLAS FP8（Marlin FP8 比 -8.6%）より高速でした。
MXFP4 の o_proj は 81.6 tok/s を達成するものの長文生成で無限ループが発生し、FP8 Marlin (80.6 tok/s) がほぼ同等速度で正常な出力を維持します。

GB10 のメモリ帯域幅は 273 GB/s です。上記量子化構成での gpt-oss-120b のアクティブ重み読み出し量は約 2.9 GB/token であり、理論上の帯域幅限界は約 93 tok/s (TP=1) です。

decode ステップで全アクティブ重みを1回読む時間を、ピーク帯域幅100%で達成できたとしたら

```
2.94 (GB/token) / 273 (GB/s) = 10.8 (ms/token)
1000 (ms) / 10.8 (ms/token) = 92.6 (token/sec)
```

実測の 62 tok/s との差は主に GEMM カーネル効率と非 GEMM 計算のオーバーヘッドによるものが考えられ、vLLM レベルのチューニングとしてはほぼ実用的な限界に近い性能ではないかと予測されます。
これ以上の大幅な高速化には、カーネル最適化、より積極的な量子化、または投機的デコード(EAGLE3 等)による1ステップあたりの生成トークン数の増加が必要になるのではないかと思われます。


## ベンチマーク結果

`vllm bench serve` (入力: 1024トークン, 出力: 128トークン, ランダムデータセット) で測定。
ハードウェア: DGX Spark (GB10 x2), ノードあたり 128 GB ユニファイドメモリ, RoCE RDMA 接続。

### Qwen3.5-35B-A3B

BF16 チェックポイント。Vanilla は BF16 推論（BF16 モデルの MXFP4 サポートなし）。
パッチ適用版は Marlin バックエンドによる BF16→MXFP4 オンライン量子化を適用。

#### 単一リクエスト (num-prompts=1, warm)

| 構成 | TPOT (ms) | スループット (tok/s) | TTFT (ms) |
|------|-----------|---------------------|-----------|
| Vanilla BF16 TP=1 | 32.93 | 28.41 | 327.34 |
| Vanilla BF16 TP=2 | 21.68 | 39.71 | 469.77 |
| **パッチ MXFP4 TP=1** | **15.18** | **60.06** | **203.52** |
| **パッチ MXFP4 TP=2** | **12.92** | **70.83** | **166.28** |

#### 10 同時リクエスト (num-prompts=10)

| 構成 | TPOT (ms) | スループット (tok/s) | 合計 tok/s |
|------|-----------|---------------------|------------|
| Vanilla BF16 TP=1 | 88.86 | 76.92 | 692.24 |
| Vanilla BF16 TP=2 | 74.77 | 91.25 | 821.24 |
| **パッチ MXFP4 TP=1** | **44.32** | **172.04** | **1548.40** |
| **パッチ MXFP4 TP=2** | **35.20** | **144.79** | **1303.07** |

### openai/gpt-oss-120b

MXFP4 事前量子化済みチェックポイント。Vanilla・パッチ適用版ともに MoE experts は MXFP4 を使用。
パッチ適用版は追加で QKV (MXFP4 Marlin)、o_proj (FP8 Marlin)、lm_head (MXFP4 Marlin) を量子化し、
SM121 Marlin MoE スレッド修正で TP=1 の正常動作を実現。

> **注意:** Vanilla vLLM 0.17.0 / 0.17.1 の TP=1 は SM121 上で Marlin MoE 256-thread カーネルの共有メモリ競合により
> ゴミ出力を生成します。パッチで修正済みです。

#### 単一リクエスト (num-prompts=1, warm)

| 構成 | TPOT (ms) | スループット (tok/s) | TTFT (ms) |
|------|-----------|---------------------|-----------|
| Vanilla MXFP4 TP=1 | -- | -- (SM121 で破損) | -- |
| Vanilla MXFP4 TP=2 | 19.09 | 48.66 | 206.48 |
| **パッチ TP=1** | **15.87** | **61.78** | **56.39** |
| **パッチ TP=2** | **12.33** | **79.28** | **48.85** |

#### 10 同時リクエスト (num-prompts=10)

| 構成 | TPOT (ms) | スループット (tok/s) | 合計 tok/s |
|------|-----------|---------------------|------------|
| Vanilla MXFP4 TP=2 | 45.79 | 175.26 | 1577.30 |
| **パッチ TP=1** | **66.02** | **148.24** | **1334.13** |
| **パッチ TP=2** | **44.15** | **181.31** | **1631.76** |

### まとめ

#### レイテンシ (単一リクエスト TPOT)

| モデル | TP | Vanilla | パッチ | 改善率 |
|--------|-----|---------|--------|--------|
| Qwen3.5-35B-A3B | 1 | 32.93 ms | **15.18 ms** | **+117%** |
| Qwen3.5-35B-A3B | 2 | 21.68 ms | **12.92 ms** | **+68%** |
| gpt-oss-120b | 1 | broken | **15.87 ms** | **修正済** |
| gpt-oss-120b | 2 | 19.09 ms | **12.33 ms** | **+55%** |

#### スループット (単一リクエスト, 出力 tok/s)

| モデル | TP | Vanilla | パッチ | 改善率 |
|--------|-----|---------|--------|--------|
| Qwen3.5-35B-A3B | 1 | 28.41 | **60.06** | **+111%** |
| Qwen3.5-35B-A3B | 2 | 39.71 | **70.83** | **+78%** |
| gpt-oss-120b | 1 | broken | **61.78** | **修正済** |
| gpt-oss-120b | 2 | 48.66 | **79.28** | **+63%** |

## パッチ内容

### vllm_all.patch (9ファイル)

| # | ファイル | 変更概要 |
|---|---------|---------|
| 1 | `vllm/envs.py` | `VLLM_MXFP4_BACKEND` 環境変数追加 (`auto`/`marlin`/`cutlass_fp4`/`triton`) |
| 2 | `vllm/.../quantization/mxfp4.py` | **主要変更**: MoE の BF16→MXFP4 オンライン量子化、QKV 用 `Mxfp4LinearMethod`、o_proj 用 `Fp8MarlinOProjLinearMethod`、lm_head 用 `Mxfp4LMHeadMethod`、`from_config()` の `modules_to_not_convert` バグ修正 |
| 3 | `vllm/.../quantization/utils/mxfp4_utils.py` | `mxfp4_e2m1_quantize()` 関数追加、expert_map assertion 除去 |
| 4 | `vllm/.../fused_moe/fused_marlin_moe.py` | **SM121 修正**: N>=2048 で w2 GEMM に 128-thread 設定を強制（256-thread カーネルの共有メモリ競合回避） |
| 5 | `vllm/.../fused_moe/layer.py` | BF16 per-expert テンソルの `weight_loader` ndim チェック修正 |
| 6 | `vllm/.../fused_moe/cutlass_moe.py` | SM121 サポート追加、EP>1 許可、expert_map 対応 |
| 7 | `vllm/.../fused_moe/routing_simulator.py` | **新規ファイル**: MoE ルーティングシミュレータ |
| 8 | `vllm/.../fla/ops/fused_recurrent.py` | Qwen3.5 の GDN (Gated Delta Net) Triton カーネル修正 |
| 9 | `vllm/.../models/gpt_oss.py` | `_quantize_moe_weight_mxfp4()` ヘルパー追加 |

### flashinfer_cutlass_sfb_layout_fix.patch

FlashInfer 0.6.4 にバンドルされた CUTLASS 4.2.1 ヘッダーの copy-paste バグ修正。
`layout_SFB` の初期化で `tile_atom_to_shape_SFA` を誤使用している問題。CUTLASS_FP4 バックエンドのみに影響。

## インストール

### 前提条件

- NVIDIA DGX Spark (GB10 / SM121)
- Python 3.12
- CUDA 13.0

### Step 1: venv 作成と vLLM インストール

```bash
python3 -m venv ~/.python-vllm-custom
source ~/.python-vllm-custom/bin/activate

pip install vllm --extra-index-url https://wheels.vllm.ai/0.17.1/cu130 \
                 --extra-index-url https://download.pytorch.org/whl/cu130
```

### Step 2: 依存パッケージのインストール・アップグレード

```bash
# NCCL アップグレード (pip 同梱の 2.28.9 は CUDAGraph + マルチノードでデッドロックするバグあり)
pip install 'nvidia-nccl-cu13>=2.29.2'

# 互換バージョンの固定
pip install 'transformers==5.3.0'
pip install 'huggingface_hub==1.5.0'

# 高速モデルロード用
pip install fastsafetensors
```

> **注意:** `huggingface_hub >= 1.6.0` では gpt-oss モデルで Harmony parser エラーが発生します。1.5.0 に固定してください。

### Step 3: パッチ適用

```bash
SITE=~/.python-vllm-custom/lib/python3.12/site-packages
cd "$SITE"

# vLLM パッチ適用 (必須)
patch -p0 < /path/to/patches/vllm_all.patch

# FlashInfer CUTLASS 修正 (CUTLASS_FP4 バックエンド使用時のみ必要)
patch -p0 < /path/to/patches/flashinfer_cutlass_sfb_layout_fix.patch
```

### Step 4: キャッシュクリア

```bash
rm -rf ~/.cache/flashinfer/
rm -rf ~/.cache/vllm/torch_compile_cache/
rm -rf ~/.cache/vllm/torch_aot_compile/
rm -rf /tmp/torchinductor_$(whoami)/
```

> **注意:** `vllm serve` の前に毎回必ずキャッシュをクリアしてください。古いキャッシュが残っていると、正常な構成でも長文生成で無限ループが発生することがあります。

### Step 5: マルチノード設定 (TP=2 のみ)

venv 全体をリモートノードに同期します:

```bash
REMOTE=<リモートノードの IP>

rsync -a ~/.python-vllm-custom/ $REMOTE:~/.python-vllm-custom/

# リモートノードのキャッシュもクリア
ssh $REMOTE "rm -rf ~/.cache/flashinfer/ ~/.cache/vllm/torch_compile_cache/ ~/.cache/vllm/torch_aot_compile/ /tmp/torchinductor_\$(whoami)/"
```

> **注意:** `vllm serve` の前に毎回必ずキャッシュをクリアしてください。古いキャッシュが残っていると、正常な構成でも長文生成で無限ループが発生することがあります。

## 実行方法

### 環境変数

```bash
# MXFP4 バックエンド (正常な出力には marlin が必須)
export VLLM_MXFP4_BACKEND=marlin
export VLLM_FASTSAFETENSORS_NOGDS=1

# 性能チューニング
export VLLM_MARLIN_USE_ATOMIC_ADD=1          # 小さい N での Marlin reduce 高速化 (SM>=90 BF16 atomicAdd)
export CUDA_DEVICE_MAX_CONNECTIONS=1          # CUDA stream 直列化で NCCL オーバーラップ改善
export PYTORCH_ALLOC_CONF=expandable_segments:True  # メモリフラグメント軽減

# マルチノード設定 (インターフェース名は環境に合わせて変更)
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

### 起動コマンド

```bash
# Qwen3.5-35B-A3B (BF16 → MXFP4 オンライン量子化, TP=2)
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

# openai/gpt-oss-120b (MXFP4 事前量子化済み, TP=2)
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


## 注意事項

- **o_proj には FP8 が必須で、MXFP4 は使えません。** E2M1 の量子化誤差が residual path を通じて 36層分累積し、長文生成で無限繰り返しループが発生します。パッチが自動的に o_proj を FP8 に設定します。
- **NCCL バージョン確認:** `pip install` のたびに NCCL が 2.28.9 にダウングレードされる可能性があります。パッケージ追加後は必ず `pip show nvidia-nccl-cu13` で確認してください。
- **キャッシュクリア:** パッチ適用時や TP サイズ変更時は、**全ノード** で torch.compile と FlashInfer のキャッシュをクリアしてください。
- **`--gpu-memory-utilization 0.80`:** GB10 のユニファイドメモリは保守的なメモリ確保が必要です。CUDAGraph capture で OOM が発生する場合は 0.70 に下げてください。
- **品質テスト:** 短文テスト (256トークン) では o_proj の精度問題を検出できません。量子化設定を変更した後は、必ず長文ストリーミング生成 (1000トークン以上, temperature=0) で検証してください。
- gpt-oss-120bの vocab の TIKTOKEN 設定などについては他のサイトをご確認ください。

## パッケージバージョン

| パッケージ | バージョン |
|-----------|-----------|
| vllm | 0.17.0+cu130 or 0.17.1+cu130 |
| torch | 2.10.0+cu130 |
| triton | 3.6.0 |
| nvidia-nccl-cu13 | >= 2.29.2 |
| fastsafetensors | 0.2.2 |
| flashinfer-python | 0.6.4 |
| transformers | 5.3.0 |
| huggingface_hub | 1.5.0 |


この成果は、SM121 の問題を特定・文書化した DGX Spark コミュニティの取り組みの上に構築されています。
参考: [DGX Spark SM121 Software Support Discussion (NVIDIA Forums)](https://forums.developer.nvidia.com/t/dgx-spark-sm121-software-support-is-severely-lacking-official-roadmap-needed/357663)

## ライセンス

これらのパッチは DGX Spark コミュニティ向けに現状のまま (as-is) で提供されます。vLLM プロジェクト本体は Apache-2.0 ライセンスです。
