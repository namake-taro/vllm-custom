#!/usr/bin/env python3
"""List quantizable layers for a HuggingFace model.

Usage:
    vllm_list_layers.py Qwen/Qwen3.5-397B-A17B-FP8
    vllm_list_layers.py openai/gpt-oss-120b
    vllm_list_layers.py --verbose Qwen/Qwen3.5-35B-A3B

Reads safetensors metadata to determine tensor shapes and classifies
layers as quantizable (2D weight tensors) or non-quantizable.
No hardcoded layer name lists - works with any model architecture.

Shows which layer names can be used with:
    VLLM_NF4_LAYERS, VLLM_MXFP4_LAYERS, VLLM_FP8_LAYERS
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def resolve_model_path(model_id: str) -> Path:
    """Resolve model_id to local path (HF cache or direct path)."""
    p = Path(model_id)
    if p.is_dir() and (p / "model.safetensors.index.json").exists():
        return p

    cache_dir = Path(os.environ.get(
        "HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    repo_dir = cache_dir / f"models--{model_id.replace('/', '--')}"
    if not repo_dir.exists():
        print(f"Error: Model not found in cache: {repo_dir}", file=sys.stderr)
        sys.exit(1)

    snapshots = repo_dir / "snapshots"
    if not snapshots.exists():
        print(f"Error: No snapshots found: {snapshots}", file=sys.stderr)
        sys.exit(1)

    snap_dirs = sorted(snapshots.iterdir(), key=lambda x: x.stat().st_mtime,
                       reverse=True)
    for snap in snap_dirs:
        if (snap / "model.safetensors.index.json").exists():
            return snap

    print(f"Error: No model.safetensors.index.json found", file=sys.stderr)
    sys.exit(1)


def get_tensor_shapes(model_path: Path,
                      weight_map: dict[str, str]) -> dict[str, list[int]]:
    """Read tensor shapes from safetensors files.

    Only reads one representative tensor per unique suffix to avoid
    opening all shard files (e.g. 397B has 94 shards, 189K tensors).
    """
    try:
        from safetensors import safe_open
    except ImportError:
        return {}

    # Pick one representative key per suffix (first occurrence)
    suffix_to_key = {}
    for key in weight_map:
        parts = key.split(".")
        # suffix = last 2 meaningful parts (e.g. "o_proj.weight")
        suffix = ".".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        if suffix not in suffix_to_key:
            suffix_to_key[suffix] = key

    # Group representative keys by shard
    shard_to_keys = defaultdict(list)
    for key in suffix_to_key.values():
        shard = weight_map[key]
        shard_to_keys[shard].append(key)

    shapes = {}  # key -> (shape, dtype_str)
    for shard, keys in shard_to_keys.items():
        shard_path = model_path / shard
        if not shard_path.exists():
            continue
        try:
            with safe_open(str(shard_path), framework="pt") as sf:
                for key in keys:
                    if key in sf.keys():
                        slc = sf.get_slice(key)
                        shapes[key] = (slc.get_shape(),
                                       str(slc.get_dtype()))
        except Exception:
            continue

    # Propagate shapes to all keys with the same suffix
    all_shapes = {}
    for key in weight_map:
        parts = key.split(".")
        suffix = ".".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        rep_key = suffix_to_key.get(suffix)
        if rep_key and rep_key in shapes:
            all_shapes[key] = shapes[rep_key]

    return all_shapes


def classify_weight(name: str, shape: list[int] | None) -> tuple[str, str, bool]:
    """Classify a weight tensor by name and shape.

    Classification rules (no hardcoded layer names):
    1. Skip scale/scale_inv tensors (quantization metadata)
    2. Skip bias tensors (1D, small)
    3. 2D tensors ending in .weight with min_dim >= 64 → quantizable
    4. Everything else → non-quantizable

    Returns: (layer_suffix, category, is_quantizable)
    """
    # Extract suffix: last meaningful component before .weight/.bias etc.
    parts = name.split(".")
    # Find the "type" suffix (strip trailing weight/bias/scale etc.)
    type_suffix = parts[-1]
    if type_suffix in ("weight", "bias"):
        type_suffix = parts[-2] if len(parts) >= 2 else type_suffix

    # Scale tensors are never directly quantizable
    if any(s in name for s in ("_scale", "scale_inv")):
        return type_suffix, "scale", False

    # Determine if this is a weight tensor
    is_weight = name.endswith(".weight")

    # Classify by shape
    if shape is not None:
        ndim = len(shape)
        if ndim == 1:
            # 1D: norm weights, biases, etc.
            return type_suffix, "norm_or_bias", False
        elif ndim >= 2 and is_weight:
            min_dim = min(shape)
            if min_dim >= 64:
                # 2D+ weight with substantial dimensions → quantizable
                # Determine subcategory from path
                is_expert = "experts" in name
                is_lm_head = "lm_head" in name
                is_embed = "embed" in name and "lm_head" not in name

                if is_embed:
                    return type_suffix, "embedding", True
                elif is_lm_head:
                    return type_suffix, "lm_head", True
                elif is_expert:
                    return type_suffix, "moe", True
                else:
                    # Could be attention proj, shared expert, dense layer, etc.
                    return type_suffix, "linear", True
            else:
                # Small dimension (e.g. gate router with dim=num_experts)
                return type_suffix, "small_linear", False
        elif ndim >= 2 and not is_weight:
            # 2D non-weight (e.g. pre-packed blocks)
            is_expert = "experts" in name
            if is_expert:
                return type_suffix, "moe_packed", True
            return type_suffix, "other", False
    else:
        # No shape info: fallback to name heuristics
        if is_weight and "norm" not in name:
            if "embed" in name and "lm_head" not in name:
                return type_suffix, "embedding", True
            return type_suffix, "linear", True
        return type_suffix, "unknown", False

    return type_suffix, "unknown", False


def main():
    parser = argparse.ArgumentParser(
        description="List quantizable layers for a HuggingFace model.")
    parser.add_argument("model", help="Model ID or local path")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show non-quantizable layers and examples")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)

    index_file = model_path / "model.safetensors.index.json"
    with open(index_file) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})

    print(f"Reading tensor shapes...", file=sys.stderr)
    shapes = get_tensor_shapes(model_path, weight_map)
    has_shapes = len(shapes) > 0
    if not has_shapes:
        print("Warning: safetensors not installed, using name heuristics.",
              file=sys.stderr)

    # Classify all weights
    quantizable = defaultdict(lambda: {"count": 0, "categories": set(),
                                        "examples": [], "shapes": set()})
    non_quantizable = defaultdict(lambda: {"count": 0, "categories": set(),
                                            "examples": [], "shapes": set()})

    for weight_name in sorted(weight_map.keys()):
        if "visual" in weight_name or "vision" in weight_name:
            continue

        shape_info = shapes.get(weight_name)
        shape = shape_info[0] if shape_info else None
        dtype_str = shape_info[1] if shape_info else None
        suffix, category, is_quant = classify_weight(weight_name, shape)

        target = quantizable if is_quant else non_quantizable
        target[suffix]["count"] += 1
        target[suffix]["categories"].add(category)
        if len(target[suffix]["examples"]) < 2:
            target[suffix]["examples"].append(weight_name)
        if shape:
            target[suffix]["shapes"].add(tuple(shape))
        if dtype_str:
            target[suffix].setdefault("dtype", dtype_str)

    def cat_str(categories):
        return "+".join(sorted(categories))

    DTYPE_BYTES = {
        "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
        "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    }

    def size_kb(shape_tuple, dtype_str=None):
        """Estimate tensor size in KB."""
        elems = 1
        for d in shape_tuple:
            elems *= d
        bpe = DTYPE_BYTES.get(dtype_str, 2)  # default BF16
        return elems * bpe / 1024

    def shape_and_size(shapes_set, dtype_str=None):
        if not shapes_set:
            return "", ""
        if len(shapes_set) == 1:
            s = list(shapes_set)[0]
            kb = size_kb(s, dtype_str)
            if kb >= 1024:
                return str(s), f"{kb/1024:.1f} MB"
            return str(s), f"{kb:.0f} KB"
        return f"{len(shapes_set)} shapes", ""

    # Print quantizable layers
    print(f"Model: {args.model}")
    print()
    print(f"{'Layer Name':<22} {'Category':<12} {'N':>5} "
          f"{'Shape':<22} Size")
    print("-" * 72)

    for suffix in sorted(quantizable.keys(),
                         key=lambda x: (cat_str(quantizable[x]["categories"]),
                                        x)):
        info = quantizable[suffix]
        dtype = info.get("dtype")
        sh, sz = shape_and_size(info["shapes"], dtype)
        print(f"{suffix:<22} {cat_str(info['categories']):<12} "
              f"{info['count']:>5} {sh:<22} {sz}")

    if args.verbose:
        # Verbose: show non-quantizable layers
        print()
        print("--- Non-quantizable ---")
        print(f"{'Layer Name':<22} {'Category':<12} {'N':>5} "
              f"{'Shape':<22} Size")
        print("-" * 72)

        for suffix in sorted(non_quantizable.keys(),
                             key=lambda x: (cat_str(
                                 non_quantizable[x]["categories"]), x)):
            info = non_quantizable[suffix]
            dtype = info.get("dtype")
            sh, sz = shape_and_size(info["shapes"], dtype)
            print(f"{suffix:<22} {cat_str(info['categories']):<12} "
                  f"{info['count']:>5} {sh:<22} {sz}")

        # Verbose: show examples
        print()
        print("Examples:")
        for suffix in sorted(quantizable.keys()):
            info = quantizable[suffix]
            print(f"  {suffix}: {info['examples'][0]}")

    # Env var suggestions
    qkv_names = {"q_proj", "k_proj", "v_proj", "qkv_proj", "in_proj_qkv"}
    o_names = {"o_proj"}

    qkv = sorted(s for s in quantizable if s in qkv_names)
    o = sorted(s for s in quantizable if s in o_names)
    moe = sorted(s for s, v in quantizable.items()
                 if "moe" in v["categories"] or "moe_packed" in v["categories"])
    lm = sorted(s for s, v in quantizable.items()
                if "lm_head" in v["categories"])

    print()
    if moe:
        print(f"# MoE+QKV=MXFP4, o_proj+lm_head=NF4:")
        print(f"  VLLM_MXFP4_LAYERS=moe,{','.join(qkv)}")
        print(f"  VLLM_NF4_LAYERS={','.join(o + lm)}")
    print(f"# All: VLLM_NF4_LAYERS=all")


if __name__ == "__main__":
    main()
