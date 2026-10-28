"""Quantize a Flash-Next pack's unquantized vision tower to EXL3, as a sidecar the loader reads.

    python tools/fn_tower_quantize.py --pack ~/flashnext-packs/unc-4.05

A pack whose vision tower shipped unquantized (16-bit, with the attention's qkv pre-fused into one
weight) does not load: the tower loader (engine/models/flashnext/vision.py) builds the tower from
EXL3 blocks, one a projection, and fuses qkv itself. This writes the tower as EXL3 blocks in one
sidecar file, vision_tower_exl3.safetensors, in the pack folder (or --out), where the loader reads
it through its vision_*.safetensors rule (engine/models/flashnext/pack.py), over the shards. The
pack's own files are not touched; deleting the sidecar puts the pack back as published.

Needs exllamav3==1.4.6 (the version the packs' quantization_config.json records) and torch, on a
CUDA box:

    pip install exllamav3==1.4.6

The treatment is the converter's own for a vision tower (exllamav3/conversion/convert_model.py):
quantize_side_model() ("uncalibrated side models: MTP head, vision tower") converts each linear
with an empty Hessian (Linear.init_H_data(False), count 0), so quantize_exl3 takes its
uncalibrated fallback path (seeded random input/output sign flips, regularize(), fallback_quant) --
the tower gets no calibration data of any kind, and the converter's bundled text corpus never
touches it either. quant_args mirror make_quant_args() with the converter's parser defaults:
codebook "mul1", --out_scales "always" (apply_out_scales=True), K = vision_bits (5 here, --bits),
seed = the side model's index (the tower is one side model, so 0 for every linear).

The layout work, from the loader's expectations:
  * the fused attn.qkv.weight [3*hidden, hidden] splits q, k, v by rows (the fused bias the same);
    the fused tensors themselves are never read by the loader;
  * mlp.linear_fc1's output and mlp.linear_fc2's input are padded with zeros to a multiple of 128
    (4304 to 4352 in this tower), the fc1 bias zero-extended with them;
  * a linear's sidecar tensors are its .trellis/.suh/.svh/.mul1 plus its bias in fp16, the tensor
    set a pack whose tower shipped quantized carries (tensor names keep the shards' model.visual.
    prefix).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

try:
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
except ImportError:
    sys.exit("exllamav3 is not importable: pip install exllamav3==1.4.6 (see this file's header)")


def h_data(k: int, key: str, dev: str) -> dict:
    # Exactly Linear.init_H_data(False) from exllamav3/modules/linear.py: an empty Hessian.
    return {
        "H": torch.zeros((k, k), dtype=torch.float32, device="meta"),
        "first_key": key,
        "count": 0,
        "finalized": False,
        "num_total": 0,
        "inf_nan": torch.zeros((2,), dtype=torch.long, device=dev),
        "device": torch.device(dev),
    }


def quantize_one(name: str, w_out_in: torch.Tensor, bits: int, dev: str):
    # quantize_exl3 takes the weight in (in_features, out_features) orientation,
    # i.e. LinearFP16.get_weight_tensor() == nn weight .T
    w = w_out_in.float().t().contiguous().to(dev)
    qargs = {
        "seed": 0,
        "K": bits,
        "devices": [dev],
        "device_ratios": None,
        "apply_out_scales": True,
        "debug_dir": None,
        "mul1": True,
    }
    _, proxy_err, out = quantize_exl3(w, h_data(w.shape[0], name, dev), qargs, False, None, False, None)
    del w
    torch.cuda.empty_cache()
    return {f"{name}.{s}": t.detach().cpu().contiguous() for s, t in out.items()}, proxy_err


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", required=True, help="the pack folder (its model.safetensors.index.json names the shards)")
    ap.add_argument("--out", default=None, help="the sidecar to write (default: <pack>/vision_tower_exl3.safetensors)")
    ap.add_argument("--bits", type=int, default=5, help="the tower's EXL3 bits a weight (default 5, the converter's vision_bits)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out_path = a.out or os.path.join(a.pack, "vision_tower_exl3.safetensors")

    idx = json.load(open(os.path.join(a.pack, "model.safetensors.index.json")))
    vis: dict[str, torch.Tensor] = {}
    for shard in sorted(set(idx["weight_map"].values())):
        with safe_open(os.path.join(a.pack, shard), framework="pt") as f:
            for k in f.keys():
                if k.startswith("model.visual."):
                    vis[k] = f.get_tensor(k)
    print(f"loaded {len(vis)} BF16 visual tensors", flush=True)

    blocks = sorted({int(k.split(".")[3]) for k in vis if k.startswith("model.visual.blocks.")})
    qkv0 = vis["model.visual.blocks.0.attn.qkv.weight"]
    hidden = qkv0.shape[1]
    assert qkv0.shape[0] == 3 * hidden, qkv0.shape

    def pad_rows(w: torch.Tensor) -> torch.Tensor:
        # the loader's shape asserts want a multiple of 128 a side
        target = -(-w.shape[0] // 128) * 128
        if target == w.shape[0]:
            return w
        out = torch.zeros((target, w.shape[1]), dtype=w.dtype)
        out[: w.shape[0]] = w
        return out

    def pad_cols(w: torch.Tensor) -> torch.Tensor:
        target = -(-w.shape[1] // 128) * 128
        if target == w.shape[1]:
            return w
        out = torch.zeros((w.shape[0], target), dtype=w.dtype)
        out[:, : w.shape[1]] = w
        return out

    def pad_bias(b: torch.Tensor, n: int) -> torch.Tensor:
        if b.shape[0] == n:
            return b
        out = torch.zeros((n,), dtype=b.dtype)
        out[: b.shape[0]] = b
        return out

    jobs = []  # (out_name, weight[out,in], bias)
    for i in blocks:
        b = f"model.visual.blocks.{i}"
        qkv_w, qkv_b = vis[f"{b}.attn.qkv.weight"], vis[f"{b}.attn.qkv.bias"]
        for j, proj in enumerate(("q_proj", "k_proj", "v_proj")):
            jobs.append((f"{b}.attn.{proj}",
                         qkv_w[j * hidden:(j + 1) * hidden].contiguous(),
                         qkv_b[j * hidden:(j + 1) * hidden].contiguous()))
        jobs.append((f"{b}.attn.proj", vis[f"{b}.attn.proj.weight"], vis[f"{b}.attn.proj.bias"]))
        fc1_w = pad_rows(vis[f"{b}.mlp.linear_fc1.weight"])
        jobs.append((f"{b}.mlp.linear_fc1", fc1_w,
                     pad_bias(vis[f"{b}.mlp.linear_fc1.bias"], fc1_w.shape[0])))
        jobs.append((f"{b}.mlp.linear_fc2", pad_cols(vis[f"{b}.mlp.linear_fc2.weight"]),
                     vis[f"{b}.mlp.linear_fc2.bias"]))
    m = "model.visual.merger"
    jobs.append((f"{m}.linear_fc1", vis[f"{m}.linear_fc1.weight"], vis[f"{m}.linear_fc1.bias"]))
    jobs.append((f"{m}.linear_fc2", vis[f"{m}.linear_fc2.weight"], vis[f"{m}.linear_fc2.bias"]))
    print(f"{len(jobs)} linears to quantize at {a.bits} bpw", flush=True)

    out_tensors: dict[str, torch.Tensor] = {}
    t0 = time.time()
    for n, (name, w, bias) in enumerate(jobs):
        tensors, perr = quantize_one(name, w, a.bits, a.device)
        out_tensors.update(tensors)
        out_tensors[f"{name}.bias"] = bias.to(torch.float16).contiguous()
        if n % 20 == 0 or n == len(jobs) - 1:
            print(f"  [{n + 1}/{len(jobs)}] {name} proxy_err={perr:.5f} elapsed={time.time() - t0:.0f}s", flush=True)
    print(f"quantized all in {time.time() - t0:.0f}s; {len(out_tensors)} tensors", flush=True)
    save_file(out_tensors, out_path, metadata={"format": "pt"})
    print(f"wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
