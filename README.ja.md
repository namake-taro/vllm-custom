# vLLM custom (for DGX Spark): STREAM LOADING

[English](README.md)

* このプロジェクトは個人的な興味や学習、研究目的のために行われているものです。研究目的などのためにご利用ください。

## なぜ STREAM LOADING か

* DGX Sparkユーザーは、新しい大型モデルが発表されたとき、4bit事前量子化されたモデルが提供されるまでなかなか試すことが難しい状況に置かれてきました。たとえそのモデルが、4bitオンザフライ量子化さえできていれば 128GiB のメモリに収まる量だったとしてもです。
* vLLM は多くのプラットフォームをターゲットとしているため、DGX Spark のメモリ構成では最大限の性能を引き出すことが非常に難しくなっています。
* 例えば、BF16の重みデータを4bit量子化する場合、デフォルトの vLLM では BF16 の全データに加えて変換後の 4bit データセットを同時にメモリ上に展開する必要があります。BF16 の段階で 128GiB に収まらないモデルは、たとえ 4bit 化後は収まるサイズであっても、起動できません。
* STREAM LOADING はこの「BF16 を一旦全部読まないと量子化できない」という制約を取り除き、ストレージから expert / layer 単位で必要分だけ読み込みながらオンザフライで 4bit 化して GPU に配置 することで、4bit 化後のメモリ量だけでモデルロードできるようにします。(残念ながら起動までの時間は大幅に増えます)


## STREAM LOADING の主機能と補助機能

### STREAM LOADING

* gpt-oss や Qwen のようにシャードがエキスパートの順に並んでいる素直なものだけではなく、Nemotron のようにシャードがエキスパート順に並んでいないモデルについても、ランダムアクセスロードによりストリーム量子化を実現しています。
* 起動時に必要なロードバッファは「同時に量子化する 1 単位分」だけで済むため、本来は GB10 1ノードに到底入らないはずの大型モデルが動作可能になります。
* 例: Qwen3.5-397B-A17B-FP8 (weights だけで約 96.7 GiB/GPU、TP=2) が DGX Spark 上で動作します。

### NF4 量子化 (MXFP4 のサブモード)

* MXFP4 (E2M1) は4bitに量子化することによって転送データ量を減らし高速化が実現できます。しかしその量子化誤差によって出力品質が悪化することが避けられません。
* そこで MXFP4 のサブモードとして、MXFP4(NF4) 量子化を実装しました。
* MXFP4 は4bit、すなわち16通りのデータに均等に量子化します。これを等幅ではなく正規分布に基づいた16分割に変更することによって、大幅に精度を向上させることができました。
* `--quantization mxfp4` の枠組みの中で起動し、レイヤーごとに `VLLM_NF4_LAYERS` 等の環境変数で MXFP4 / NF4 / FP8 を切り替えることができます。

### 自動 KV cache 確保

* これまで、DGX Sparkユーザーは KV cache のメモリを最大化するために `--gpu-memory-utilization` を 0.1 ずつずらしながら試行錯誤して最適値を探してきました。
* STREAM LOADING で巨大モデルが動くようになると、weights が空きメモリを圧迫するため、この手動チューニングはさらに困難になります。
* そこで `auto` (デフォルト値) を指定することによって、その時の空きメモリの最大限で KV cache サイズを確保するように修正しました。
* まず最小限の KV cache 確保を行い、torch.compile や flashinfer JIT の後に、KV cache を解放してメモリ量を再計算した後に、限界値まで KV cache を再確保します。caching allocator のフラグメンテーションプールも考慮して計算するため、推論中の OOM を回避します。
* 推論で使用するメモリ量に合わせて、メモリ限界量からのマージンをユーザーが環境変数で設定することも可能です。


## ベンチマーク結果

* 下記のいくつかは TP=1 または TP=2 では本来絶対に実行できないはずのものが入っていることがわかるでしょう。(下記のモデルはInt4やNVFP4などの4bit事前量子化モデルではありません。)
* この結果はレイヤー指定などの多くをデフォルト値にしてあります。更なるチューニングが可能だと思われます。
* スループットは llama-benchy の出力です。

### スループット

* Decode スループット 1並列 (tg128, c1, tokens/s)

| モデル | TP=1 | TP=2 |
|--------|------|------|
| gpt-oss-120b | 64.52 | **79.55** |
| Qwen3.5-35B-A3B | 64.37 | **78.45** |
| Qwen3.5-27B | 12.07 | **20.46** |
| Qwen3.5-122B-A10B | 28.17 | **41.90** |
| Nemotron3-120B-A12B-BF16 | 24.11 | **36.58** |
| Qwen3.5-397B-A17B-FP8 | - | **26.83** |

* Decode スループット 10並列 (tg128, c10, tokens/s, total)

| モデル | TP=1 | TP=2 |
|--------|------|------|
| gpt-oss-120b | 165.02 | 198.19 |
| Qwen3.5-35B-A3B | 208.31 | 161.37 |
| Qwen3.5-27B | 92.54 | 95.18 |
| Qwen3.5-122B-A10B | 74.90 | 75.11 |
| Nemotron3-120B-A12B-BF16 | 84.56 | 61.48 |
| Qwen3.5-397B-A17B-FP8 | - | 50.98 |


### KV cache size

| モデル | TP=1 KV cache (tokens) | TP=1 max concurrency | TP=2 KV cache (tokens) | TP=2 max concurrency |
|--------|------------------------|----------------------|------------------------|----------------------|
| gpt-oss-120b | 962,352 | 11.74x | 3,395,392 | 41.41x |
| Qwen3.5-35B-A3B | 1,852,864 | 26.80x | 4,106,064 | 59.38x |
| Qwen3.5-27B | 500,192 | 7.33x | 1,204,224 | 17.67x |
| Qwen3.5-122B-A10B | 538,704 | 7.48x | 2,405,376 | 33.41x |
| Nemotron3-120B-A12B-BF16 | 772,992 | 11.44x | 4,456,320 | 65.99x |
| Qwen3.5-397B-A17B-FP8 | - | - | 91,872 | 2.32x |

* デフォルト値はOOM安全のため1GiBの猶予を持たせてあります。

* `max concurrency` は `max_model_len` token の prompt が同時に何件入るかの倍率。


## インストール方法

### 前提

- NVIDIA DGX Spark (GB10 / SM121)
- Python 3.12
- CUDA 13.0
- (TP=2 の場合) DGX Spark 2 ノード + マルチノード通信設定
- vLLM custom (vllm 0.17.1 ベース)

### Step 1: venv 作成

```bash
python3 -m venv ~/.python-vllm-custom
source ~/.python-vllm-custom/bin/activate
```

### Step 2: vLLM custom のインストール

```bash
# vLLM 0.17.1 ベースの custom wheel
pip install --extra-index-url https://download.pytorch.org/whl/cu130 \
  https://github.com/namake-taro/vllm-custom/releases/download/v0.17.1+sparkcustom.00efb251a6/vllm-0.17.1+sparkcustom.00efb251a6.precompiled-cp312-cp312-linux_aarch64.whl
pip install 'nvidia-nccl-cu13>=2.29.2' > /dev/null 2>&1
```

> 全ての修正ソースのパッケージは [Releases ページ](https://github.com/namake-taro/vllm-custom/releases) に `.src.tar.gz` として置いてあります。

### Step 3: キャッシュクリア

```bash
rm -rf ~/.cache/flashinfer/
rm -rf ~/.cache/vllm/torch_compile_cache/
rm -rf ~/.cache/vllm/torch_aot_compile/
rm -rf /tmp/torchinductor_$(whoami)/
```

> 注意: `vllm serve` の前に毎回必ずキャッシュをクリアしてください。古いキャッシュが残っていると、正常な構成でも長文生成で無限ループが発生することがあります。


## 実行方法、環境設定方法

* BF16, FP8 モデルの多くにおいては、4bit 量子化分が最終的にメモリに入りさえすれば、`vllm serve` のオプションとして `--quantization mxfp4` を追加するだけで、デフォルト設定か僅かな変更で実行できるでしょう。`--gpu-memory-utilization` はもはや設定する必要はありません（デフォルトが auto です）。

### 環境変数

```bash
# === STREAM LOADING の動作に必須 ===

# ストリームローディングの有効・無効を変更できます (ON=1, OFF=0)
export VLLM_STREAM_LOADING=1

# GB10 は GDS (GPUDirect Storage) に非対応のため、1 に設定してください
export VLLM_FASTSAFETENSORS_NOGDS=1

# MXFP4 バックエンド (正常な出力には marlin が必須)
export VLLM_MXFP4_BACKEND=marlin


# レイヤーごとに量子化方法を切り替えることができます (コンマ区切りで複数指定できます)
# 優先順位は FP8 > NF4 > MXFP4 です
# all を指定するとすべての量子化対象レイヤーに適用されます
export VLLM_FP8_LAYERS=
export VLLM_NF4_LAYERS=all
export VLLM_MXFP4_LAYERS=

# NF4 量子化のグループサイズを変更できます
# 32, 64, 128, 256, 512 (32 が一番出力品質が良いです)
export VLLM_NF4_GROUP_SIZE=128

# KV cache を確保するときのメモリ上限値からの空きメモリ分を指定します (MiB)
# デフォルト値は安全のため猶予を持たせてあります。
# ただしマージンを小さくしすぎると llama-benchy の実行でも OOM が起きる場合があります
export VLLM_KV_CACHE_MEM_MARGIN=1024

# その他の設定例
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
export NCCL_NET_GDR_LEVEL=5
export NCCL_PROTO=LL,LL128,Simple
export NCCL_SOCKET_IFNAME=enp1s0f1np1
```


## 実行コマンド例

### openai/gpt-oss-120b (TP=1) の例

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

### Qwen/Qwen3.5-397B-A17B-FP8 (TP=2) の例

> **注:** 397B のように weights が片ノードあたり 96 GiB を超える極限ケースでは、warmup の一時メモリ消費を最小化するため `--max-num-batched-tokens` と `--max-cudagraph-capture-size` を小さく設定する必要があります。他の小型モデルでは `32768` / `32` 程度の通常値で構いません。

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


## レイヤー指定について

### レイヤー情報表示簡易コマンド

* リポジトリに同梱の `tools/list_layers.py` を使ってモデルの量子化対象レイヤーを確認できます。

```
./tools/list_layers.py openai/gpt-oss-120b
```

openai/gpt-oss-120b のレイヤー情報が得られます。

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

### レイヤー情報に基づいた環境変数の設定例

* 例えば、gpt-oss-120b においては、上の出力結果を参考に次のような設定が可能です。
* 優先順位は FP8 > NF4 > MXFP4 です。

```bash
export VLLM_FP8_LAYERS=o_proj,lm_head
export VLLM_NF4_LAYERS=moe,q_proj,k_proj,v_proj
export VLLM_MXFP4_LAYERS=all
```


## 成果の帰属

* この成果は、DGX Spark, GB10, SM121 の問題を特定・文書化した NVIDIA Developer Forum DGX Spark コミュニティの取り組みの上に構築されています。
参考: [Computing/DGX Spark / GB10 User Forum topics - NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/)


## ライセンス

* これらのソースは DGX Spark コミュニティ向けに現状のまま (as-is) で提供されます。vLLM プロジェクト本体は Apache-2.0 ライセンスです。
