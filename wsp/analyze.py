"""Error anatomy + the key weight-space geometry for the 'word splitting' failure."""
import json
import torch
import jiwer
from common import ROOT, load_model

pairs = json.loads((ROOT / "out" / "baseline_dev.json").read_text(encoding="utf-8"))["pairs"]

# 1) how much WER is pure segmentation (spaces in wrong place)?
splits = merges = 0
for p in pairs:
    r, h = p["ref"].split(), p["hyp"].split()
    rs = set(r)
    for a, b in zip(h, h[1:]):
        if a + b in rs and a not in rs:
            splits += 1
    hs = set(h)
    for a, b in zip(r, r[1:]):
        if a + b in hs:
            merges += 1
refs = [p["ref"] for p in pairs]; hyps = [p["hyp"] or "x" for p in pairs]
print(f"WER={jiwer.wer(refs, hyps):.4f}")
print(f"wrongly SPLIT words: {splits}   wrongly MERGED words: {merges}   total ref words: {sum(len(r.split()) for r in refs)}")
nospace = jiwer.cer([r.replace(' ', '') for r in refs], [h.replace(' ', '') for h in hyps])
print(f"CER with spaces={jiwer.cer(refs, hyps):.4f}  CER ignoring spaces={nospace:.4f}")

# 2) geometry: output logits = LN(h) @ E^T  (proj_out is tied to embed_tokens)
proc, model = load_model()
tok = proc.tokenizer
E = model.model.decoder.embed_tokens.weight.detach().float().cpu()
print("tied proj_out:", model.proj_out.weight.data_ptr() == model.model.decoder.embed_tokens.weight.data_ptr())
vocab = tok.get_vocab()
sp, ns = [], []
for t, i in vocab.items():
    if i >= tok.eos_token_id:  # skip special tokens
        continue
    s = tok.convert_tokens_to_string([t])
    core = s.strip()
    if not core.isalpha():
        continue
    (sp if s.startswith(" ") else ns).append(i)
sp, ns = torch.tensor(sp), torch.tensor(ns)
d = E[ns].mean(0) - E[sp].mean(0)
proj = E @ (d / d.norm())
print(f"space-prefixed tokens: {len(sp)}  no-space (suffix) tokens: {len(ns)}")
print(f"|d|={d.norm():.3f}  proj(no-space) mean={proj[ns].mean():+.3f}  proj(space) mean={proj[sp].mean():+.3f}  std={proj.std():.3f}")
lnb = model.model.decoder.layer_norm.bias.detach().float().cpu()
print(f"decoder final LN bias: norm={lnb.norm():.3f}; its current push along suffix direction: {float(lnb @ (d/d.norm())):+.3f}")
