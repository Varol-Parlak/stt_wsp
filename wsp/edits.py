"""Weight-space edits for Turkish Whisper. Each edit is a closed-form, hand-designed change to specific
weights (no gradients, no training loop). We sweep its strength on DEV and keep it only if DEV improves.

Usage: python edits.py sweep            # try all edits/strengths on dev
       python edits.py final A=2 B=1    # apply chosen edits, eval dev+test, save weights
"""
import sys
import json
import copy
from collections import Counter

import torch
from safetensors.torch import save_file
from common import ROOT, load_model, load_split, evaluate, fmt

OUT = ROOT / "out"
proc, base = load_model()
tok = proc.tokenizer
LOG = OUT / "edits_log.jsonl"


# ---------------- Edit A: Turkish lexical prior written into decoder.layer_norm.bias ----------------
# logits_t = (gamma*norm(h) + beta) . E_t   (proj_out tied to embed_tokens)
# => adding delta to beta adds (delta . E_t) to token t's logit at EVERY step: a global, learnable-free prior.
def turkish_prior_delta():
    E = base.model.decoder.embed_tokens.weight.detach().float().cpu()
    V = E.shape[0]
    text = (ROOT / "tr_corpus.txt").read_text(encoding="utf-8").splitlines()
    cnt = Counter()
    for line in text:
        for variant in (line, line.lower()):
            cnt.update(tok.encode(" " + variant.strip(), add_special_tokens=False))
    n_text = tok.eos_token_id  # ids >= eos are special tokens: leave them alone
    N = sum(cnt.values())
    o = torch.zeros(V)
    for t in range(n_text):
        o[t] = torch.log(torch.tensor((cnt.get(t, 0) + 0.5) / (N + 0.5 * n_text)))
    o[:n_text] -= o[:n_text].mean()
    W = torch.zeros(V); W[:n_text] = 1.0
    Es = E * W[:, None].sqrt()
    r = 1e-3 * (Es.T @ Es).trace() / E.shape[1]
    delta = torch.linalg.solve(Es.T @ Es + r * torch.eye(E.shape[1]), Es.T @ (o * W.sqrt()))
    shift = E[:n_text] @ delta
    delta = delta / shift.std()  # unit = 1 logit std of induced shift
    seen = torch.tensor([cnt.get(t, 0) > 0 for t in range(n_text)])
    print(f"[A] corpus tokens={N} distinct={len(cnt)}; induced shift seen={float((E[:n_text]@delta)[seen].mean()):+.3f} "
          f"unseen={float((E[:n_text]@delta)[~seen].mean()):+.3f} (logits per unit strength)")
    return delta


def edit_A(model, lam, _cache={}):
    if "d" not in _cache:
        _cache["d"] = turkish_prior_delta()
    b = model.model.decoder.layer_norm.bias
    b.data += (lam * _cache["d"]).to(b.device, b.dtype)
    return 512


# ---------------- Edit B: sharpen the <|tr|> language token ----------------
LANGS = [l.strip("<|>") for l in tok.additional_special_tokens if len(l) <= 7 and l.startswith("<|") and l[2:-2].isalpha()]


def edit_B(model, beta):
    emb = model.model.decoder.embed_tokens.weight
    tr = tok.convert_tokens_to_ids("<|tr|>")
    ids = [tok.convert_tokens_to_ids(f"<|{l}|>") for l in LANGS if l != "tr"]
    mu = emb.data[ids].mean(0)
    emb.data[tr] += beta * (emb.data[tr] - mu)
    return 512


EDITS = {"A": edit_A, "B": edit_B}


def apply(spec):
    m = copy.deepcopy(base)
    n = 0
    for k, v in spec.items():
        n += EDITS[k](m, v)
    return m, n


def run(spec, samples, tag):
    m, n = apply(spec)
    r = evaluate(proc, m, samples, tag)
    r["spec"] = spec; r["params_edited"] = n
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(r) + "\n")
    print(fmt(r), f"edited={n}", flush=True)
    return m, r


if __name__ == "__main__":
    dev = load_split("dev")
    mode = sys.argv[1]
    if mode == "sweep":
        run({}, dev, "dev base")
        for lam in (0.5, 1, 2, 3, 4):
            run({"A": lam}, dev, f"dev A={lam}")
        for beta in (0.25, 0.5, 1.0, 2.0):
            run({"B": beta}, dev, f"dev B={beta}")
    elif mode == "final":
        spec = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in sys.argv[2:]}
        m, _ = run(spec, dev, f"dev {spec}")
        run(spec, load_split("test"), f"TEST {spec}")
        save_file({k: v.contiguous() for k, v in m.state_dict().items()
                   if k != "proj_out.weight"}, str(OUT / "whisper-base-tr-wsp.safetensors"))
        print("saved", OUT / "whisper-base-tr-wsp.safetensors")
