"""Weight-space inspection tools for Whisper. Prints a compact, LLM-readable 'map' of the weights.

Usage:  python inspect_weights.py                 # global map
        python inspect_weights.py <param-name>    # detail of one tensor
"""
import sys
import torch
from common import load_model

torch.set_printoptions(precision=4, sci_mode=False, linewidth=160)
proc, model = load_model()
sd = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}


def stats(t):
    f = t.flatten()
    return (f"shape={tuple(t.shape)} mean={f.mean():+.4f} std={f.std():.4f} "
            f"absmax={f.abs().max():.3f} kurt={(((f - f.mean()) / (f.std() + 1e-8)) ** 4).mean():.1f}")


if len(sys.argv) > 1:
    name = sys.argv[1]
    t = sd[name]
    print(name, stats(t))
    if t.ndim == 2:
        rn = t.norm(dim=1)
        cn = t.norm(dim=0)
        print("row norms   top10:", rn.topk(10))
        print("col norms   top10:", cn.topk(10))
        s = torch.linalg.svdvals(t)
        print("singular values (first 16):", s[:16])
        print("effective rank (99% energy):", int(((s**2).cumsum(0) / (s**2).sum() < 0.99).sum()) + 1, "/", len(s))
    else:
        print(t[:64])
    sys.exit()

# ---- global map ----
print(f"total params {sum(v.numel() for v in sd.values())/1e6:.1f}M\n")
for k, v in sd.items():
    if v.ndim >= 1 and v.numel() > 1:
        print(f"{k:70s} {stats(v)}")

# ---- language-token embeddings: where does <|tr|> sit? ----
tok = proc.tokenizer
emb = sd["model.decoder.embed_tokens.weight"]
langs = ["en", "tr", "az", "de", "fr", "es", "ru", "ar", "fa", "uz", "kk", "tk", "it", "pt", "nl", "ja"]
ids = {l: tok.convert_tokens_to_ids(f"<|{l}|>") for l in langs}
E = torch.stack([emb[i] for i in ids.values()])
En = E / E.norm(dim=1, keepdim=True)
print("\nlanguage-token embedding norms:", {l: round(float(E[j].norm()), 3) for j, l in enumerate(langs)})
sim = En @ En.T
j = langs.index("tr")
print("cos(<|tr|>, other langs):", {l: round(float(sim[j, k]), 3) for k, l in enumerate(langs) if l != "tr"})
