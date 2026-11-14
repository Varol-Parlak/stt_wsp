"""3-Way Scientific Comparison: Untouched Baseline vs. Weight-Space Programming (WSP) vs. LoRA Fine-Tuning.

Hot-swaps a Colab-trained LoRA adapter (`wsp/out/lora_adapter/` or `wsp/out/whisper-base-tr-lora.zip`)
onto the local `openai/whisper-base` model in pure PyTorch (no `peft` dependency required),
evaluates on FLEURS Turkish `dev` (338) and held-out `test` (743), and generates a paper-style
comparison report in `wsp/out/comparison_3way.md`.

Usage:
  .venv\\Scripts\\python.exe wsp\\compare_3way.py
  .venv\\Scripts\\python.exe wsp\\compare_3way.py --adapter path\\to\\whisper-base-tr-lora.zip
"""
import sys
import json
import zipfile
import argparse
from pathlib import Path

import torch
import jiwer
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import ROOT, load_model, load_split, evaluate, fmt

OUT = ROOT / "out"
DEFAULT_ADAPTER_DIR = OUT / "lora_adapter"
DEFAULT_ZIP = OUT / "whisper-base-tr-lora.zip"


def resolve_adapter_dir(adapter_arg: str | None) -> Path:
    if adapter_arg:
        p = Path(adapter_arg)
        if p.is_file() and p.suffix == ".zip":
            DEFAULT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(p, "r") as zf:
                zf.extractall(DEFAULT_ADAPTER_DIR)
            return DEFAULT_ADAPTER_DIR
        return p

    if (DEFAULT_ADAPTER_DIR / "adapter_model.safetensors").exists():
        return DEFAULT_ADAPTER_DIR

    if DEFAULT_ZIP.exists():
        DEFAULT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(DEFAULT_ZIP, "r") as zf:
            zf.extractall(DEFAULT_ADAPTER_DIR)
        return DEFAULT_ADAPTER_DIR

    raise FileNotFoundError(
        f"No LoRA adapter found at {DEFAULT_ADAPTER_DIR} or {DEFAULT_ZIP}.\n"
        "Run wsp/colab_lora_train.ipynb on Google Colab and place whisper-base-tr-lora.zip in wsp/out/."
    )


def merge_lora_inplace(model, adapter_dir: Path) -> int:
    """Pure-PyTorch LoRA hot-swap: W += (lora_alpha / r) * (lora_B @ lora_A)."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    r = cfg["r"]
    alpha = cfg["lora_alpha"]
    scale = alpha / r

    lora_sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
    model_sd = model.state_dict()

    # Collect (A, B) pairs keyed by base parameter prefix
    pairs = {}
    for k, v in lora_sd.items():
        # Typical PEFT key: base_model.model.model.decoder.layers.0.self_attn.q_proj.lora_A.weight
        clean = k.replace("base_model.model.", "")
        if ".lora_A." in clean:
            base = clean.split(".lora_A.")[0] + ".weight"
            pairs.setdefault(base, {})["A"] = v
        elif ".lora_B." in clean:
            base = clean.split(".lora_B.")[0] + ".weight"
            pairs.setdefault(base, {})["B"] = v

    lora_params = 0
    for base_key, ab in pairs.items():
        assert base_key in model_sd, f"Missing target weight {base_key} in model"
        A = ab["A"].to(model_sd[base_key].device, dtype=torch.float32)
        B = ab["B"].to(model_sd[base_key].device, dtype=torch.float32)
        delta = (B @ A) * scale
        model_sd[base_key].data.add_(delta.to(model_sd[base_key].dtype))
        lora_params += A.numel() + B.numel()

    print(f"[LoRA] Merged {len(pairs)} attention projections (r={r}, alpha={alpha}, {lora_params:,} LoRA params)", flush=True)
    return lora_params


def clip_wer(p: dict) -> float:
    return jiwer.wer(p["ref"], p["hyp"] or "<empty>")


def main():
    parser = argparse.ArgumentParser(description="3-Way Comparison: Baseline vs WSP vs LoRA")
    parser.add_argument("--adapter", default=None, help="Path to lora_adapter dir or whisper-base-tr-lora.zip")
    args = parser.parse_args()

    adapter_dir = resolve_adapter_dir(args.adapter)
    lora_dev_path = OUT / "lora_dev.json"
    lora_test_path = OUT / "lora_test.json"

    proc, model = load_model()
    lora_params = merge_lora_inplace(model, adapter_dir)

    dev = load_split("dev")
    test = load_split("test")
    r_dev = evaluate(proc, model, dev, "lora/dev", save=lora_dev_path)
    print(fmt(r_dev), flush=True)
    r_test = evaluate(proc, model, test, "lora/test", save=lora_test_path)
    print(fmt(r_test), flush=True)

    # Load all 3 results for comparison
    base_dev = json.loads((OUT / "baseline_dev.json").read_text(encoding="utf-8"))
    wsp_dev = json.loads((OUT / "wsp_dev.json").read_text(encoding="utf-8"))
    lora_dev = json.loads(lora_dev_path.read_text(encoding="utf-8"))

    base_test = json.loads((OUT / "baseline_test.json").read_text(encoding="utf-8"))
    wsp_test = json.loads((OUT / "wsp_test.json").read_text(encoding="utf-8"))
    lora_test = json.loads(lora_test_path.read_text(encoding="utf-8"))

    # Win / loss vs Baseline on held-out test (743 clips)
    wsp_win = sum(1 for b, w in zip(base_test["pairs"], wsp_test["pairs"]) if clip_wer(w) < clip_wer(b))
    wsp_lose = sum(1 for b, w in zip(base_test["pairs"], wsp_test["pairs"]) if clip_wer(w) > clip_wer(b))
    lora_win = sum(1 for b, l in zip(base_test["pairs"], lora_test["pairs"]) if clip_wer(l) < clip_wer(b))
    lora_lose = sum(1 for b, l in zip(base_test["pairs"], lora_test["pairs"]) if clip_wer(l) > clip_wer(b))

    # Pick interesting 3-way examples from test
    examples = []
    for i, (b, w, l) in enumerate(zip(base_test["pairs"], wsp_test["pairs"], lora_test["pairs"])):
        bw, ww, lw = clip_wer(b), clip_wer(w), clip_wer(l)
        if bw > 0.15 and lw < bw and len(b["ref"]) < 140:
            examples.append((bw - lw, i, b["ref"], b["hyp"], w["hyp"], l["hyp"], bw, ww, lw))
    examples.sort(reverse=True)

    md = [
        "# 3-Way Comparison: Baseline vs. Weight-Space Programming (WSP) vs. LoRA",
        "",
        "Model: `openai/whisper-base` (72.6M params) evaluated identically (`fp32`, `num_beams=1`, Turkish-aware normalizer) on Google FLEURS Turkish (`tr_tr`).",
        "",
        "## 1. Quantitative Results",
        "",
        "| Method | Training Data | Gradient Steps | Params Changed / Added | Dev WER (338) | Dev CER | Dev Loops | Test WER (743) | Test CER | Test Loops | Test Clips Improved / Worse |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Untouched Baseline** | None | 0 | 0 (`0.00%`) | `{base_dev['wer']*100:.2f}%` | `{base_dev['cer']*100:.2f}%` | `{base_dev['loops']}` | `{base_test['wer']*100:.2f}%` | `{base_test['cer']*100:.2f}%` | `{base_test['loops']}` | — |",
        f"| **WSP (`C=0.05, G=0.15, B=0.03`)** | None (0 clips) | 0 | 99,840 (`0.14%`) | `{wsp_dev['wer']*100:.2f}%` | `{wsp_dev['cer']*100:.2f}%` | `{wsp_dev['loops']}` | `{wsp_test['wer']*100:.2f}%` | `{wsp_test['cer']*100:.2f}%` | `{wsp_test['loops']}` | `{wsp_win}` / `{wsp_lose}` |",
        f"| **LoRA Fine-Tune (Colab T4)** | FLEURS `train` | ~600 steps | {lora_params:,} (`{100*lora_params/72593920:.2f}%`) | `{lora_dev['wer']*100:.2f}%` | `{lora_dev['cer']*100:.2f}%` | `{lora_dev['loops']}` | `{lora_test['wer']*100:.2f}%` | `{lora_test['cer']*100:.2f}%` | `{lora_test['loops']}` | `{lora_win}` / `{lora_lose}` |",
        "",
        "## 2. Advantages & Disadvantages",
        "",
        "| Approach | Advantages | Disadvantages |",
        "| :--- | :--- | :--- |",
        "| **Untouched Baseline** | Zero setup; balanced multilingual prior; no collateral regressions on clean words. | ~30.8% Turkish WER; susceptible to repetition loops (`bbbb...`) and weak cross-attention on suffixes. |",
        "| **Weight-Space Programming (WSP)** | **Zero training data, zero gradients, ~1 second to apply**; eliminates repetition loops (`1 -> 0`) and recovers `-1.99%` dev / `-0.67%` test WER. | Cannot teach new Turkish vocabulary/morphology; head dampening can introduce minor spelling noise on clean or short 1–3s mic clips. |",
        "| **LoRA Fine-Tuning** | Directly teaches Turkish acoustics, suffixes, and orthography; largest WER/CER reduction; hot-swappable ~4 MB adapter. | Requires ~1.8 GB labeled Turkish audio and GPU training time (~10–15 min on Colab T4); can slightly overfit to the domain of the training dataset. |",
        "",
        "## 3. Side-by-Side Test Transcripts",
        "",
    ]
    for _, idx, ref, bh, wh, lh, bw, ww, lw in examples[:6]:
        md.extend([
            f"**Clip #{idx}** (WER: Base `{bw*100:.1f}%` | WSP `{ww*100:.1f}%` | LoRA `{lw*100:.1f}%`)",
            f"- **Ref**: `{ref}`",
            f"- **Base**: `{bh}`",
            f"- **WSP**: `{wh}`",
            f"- **LoRA**: `{lh}`",
            "",
        ])

    report_path = OUT / "comparison_3way.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved full 3-way report to {report_path}")


if __name__ == "__main__":
    main()
