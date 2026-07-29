#!/usr/bin/env python3
"""
Fetches band order + normalization constants for a BIFOLD-BigEarthNetv2-0
model (e.g. BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0) by:

  1. Downloading config.json from the HF model repo (channels, image_size).
  2. Downloading configilm's BENv2_utils.py source (STANDARD_BANDS + means/stds)
     from the official ConfigILM GitHub repo.
  3. Resolving the band order for the model's channel count.
  4. Resolving the matching mean/std interpolation table for the model's image_size.

Usage:
    python get_ben_norm_config.py [model_id]

    model_id defaults to "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0"
"""

import sys
import json
import urllib.request

HF_CONFIG_URL = "https://huggingface.co/{model_id}/raw/main/config.json"
BENV2_UTILS_URL = "https://raw.githubusercontent.com/lhackel-tub/ConfigILM/main/configilm/extra/BENv2_utils.py"


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def fetch_model_config(model_id: str) -> dict:
    url = HF_CONFIG_URL.format(model_id=model_id)
    return json.loads(fetch_text(url))


def fetch_benv2_constants():
    """
    Pulls STANDARD_BANDS, means, stds out of ConfigILM's BENv2_utils.py
    by parsing just the top-level literal assignments (no lmdb/torch/pandas
    imports needed for the constants themselves).
    """
    src = fetch_text(BENV2_UTILS_URL)

    import ast

    tree = ast.parse(src)
    wanted = {"STANDARD_BANDS", "means", "stds", "_s2_bandnames",
              "_s1_bandnames", "_all_bandnames"}

    # Some assignments (e.g. STANDARD_BANDS, _all_bandnames) reference
    # earlier names or use list concatenation (e.g. _s1_bandnames + _s2_...),
    # so ast.literal_eval alone can't handle them. Instead, evaluate each
    # top-level assignment's RHS in order against a namespace that only
    # contains prior top-level names/literals from this same file — no
    # imports, calls, or other code are executed.
    namespace: dict = {}
    allowed_node_types = (
        ast.List, ast.Tuple, ast.Dict, ast.Constant, ast.Name,
        ast.BinOp, ast.Add, ast.Load, ast.Set,
        ast.UnaryOp, ast.USub,
    )

    def _is_safe(node) -> bool:
        for n in ast.walk(node):
            if not isinstance(n, allowed_node_types):
                return False
        return True

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if _is_safe(node.value):
                try:
                    value = eval(compile(ast.Expression(node.value), "<benv2_utils>", "eval"), {}, namespace)
                    namespace[name] = value
                except Exception:
                    continue

    collected = {k: v for k, v in namespace.items() if k in wanted}
    missing = wanted - collected.keys()
    if missing:
        raise RuntimeError(f"Could not extract expected constants from BENv2_utils.py: {missing}")
    return collected


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0"

    print(f"Model: {model_id}")
    print("-" * 60)

    cfg = fetch_model_config(model_id)
    channels = cfg["channels"]
    image_size = cfg["image_size"]
    print(f"channels (from config.json)   : {channels}")
    print(f"image_size (from config.json) : {image_size}")

    consts = fetch_benv2_constants()
    standard_bands = consts["STANDARD_BANDS"]
    means = consts["means"]
    stds = consts["stds"]

    if channels not in standard_bands:
        print(f"\nNo predefined STANDARD_BANDS entry for channels={channels}.")
        print(f"Available keys: {list(standard_bands.keys())}")
        return

    band_order = standard_bands[channels]
    print(f"\nBand order for channels={channels}:")
    for i, b in enumerate(band_order):
        print(f"  [{i}] {b}")

    # pick interpolation table matching image_size; fall back to *_nearest
    interp_key = f"{image_size}_nearest"
    if interp_key not in means:
        candidates = [k for k in means if k.startswith(f"{image_size}_")]
        interp_key = candidates[0] if candidates else "no_interpolation"
    print(f"\nUsing normalization table: '{interp_key}'")

    print(f"\n{'band':6}{'mean':>14}{'std':>14}")
    band_means, band_stds = [], []
    for b in band_order:
        m = means[interp_key][b]
        s = stds[interp_key][b]
        band_means.append(m)
        band_stds.append(s)
        print(f"{b:6}{m:14.4f}{s:14.4f}")

    print("\nAs Python lists (in model input order):")
    print("MEAN =", [round(m, 4) for m in band_means])
    print("STD  =", [round(s, 4) for s in band_stds])


if __name__ == "__main__":
    main()
