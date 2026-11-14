"""Weight-Space Programming (WSP) interventions for Turkish Whisper (openai/whisper-base).

Each edit is a closed-form, training-free, gradient-free surgical modification to specific weight tensors:
  - Edit A (512 params):   Turkish lexical prior projected into `decoder.layer_norm.bias` (orthogonal to EOS).
  - Edit B (512 params):   Contrastive sharpening of the `<|tr|>` language token in `embed_tokens.weight`.
  - Edit C (1024 params):  Acoustic cross-attention gain via `encoder.layer_norm` (weight & bias) to cure loops.
  - Edit D (512 params):   Layer-5 cross-attention vs self-attention LayerNorm balance.
  - Edit E (~35k params):  Nullspace logit calibration along u = 1/gamma for Turkish bound suffixes (fixes word-splitting).
  - Edit F (~80k params):  Nullspace logit suppression along u = 1/gamma for non-Turkish alphabet tokens (q, w, x, foreign).

Usage:
  python edits.py sweep                 # sweep individual edits & combinations on dev (338 clips)
  python edits.py final C=0.05 E=0.8    # apply best spec, evaluate on dev + held-out test (743 clips), save weights
"""
import sys
import json
from pathlib import Path
from collections import Counter

import torch
from safetensors.torch import save_file
from common import ROOT, load_model, load_split, evaluate, fmt

OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)
LOG = OUT / "edits_log.jsonl"

proc, model = load_model()
tok = proc.tokenizer
VOCAB = tok.get_vocab()

# Snapshot original parameters that any edit might touch (fast in-place restore, zero deepcopy overhead)
TOUCHED_KEYS = [
    "model.decoder.layer_norm.weight",
    "model.decoder.layer_norm.bias",
    "model.decoder.embed_tokens.weight",
    "model.encoder.layer_norm.weight",
    "model.encoder.layer_norm.bias",
    "model.decoder.layers.3.encoder_attn.out_proj.weight",
    "model.decoder.layers.4.self_attn.out_proj.weight",
    "model.decoder.layers.5.encoder_attn.out_proj.weight",
    "model.decoder.layers.5.self_attn.out_proj.weight",
]
ORIG = {k: model.state_dict()[k].detach().clone() for k in TOUCHED_KEYS}


def restore():
    sd = model.state_dict()
    for k, v in ORIG.items():
        sd[k].copy_(v)


# ---------------- Helper: LayerNorm exact nullspace direction u = (1 / gamma) ----------------
# For any hidden state h_t, LayerNorm_0(h_t) has zero mean across the 512 channels, so:
#   (gamma * LayerNorm_0(h_t)) . (1 / gamma) == sum_i LayerNorm_0(h_t)_i == 0  (identically for all h_t!)
# Thus shifting a token embedding E_v by (delta / (beta . u)) * u shifts token v's output logit by
# EXACTLY +delta at every step without perturbing the acoustic/contextual dot product (gamma * h_hat) . E_v.
def nullspace_unit_vec():
    g = ORIG["model.decoder.layer_norm.weight"].float().cpu()
    b = ORIG["model.decoder.layer_norm.bias"].float().cpu()
    u = 1.0 / g
    u = u / u.norm()
    bu = float(b @ u)
    return u, bu


# ---------------- Edit A: Turkish lexical prior in decoder.layer_norm.bias (512 params) ----------------
def turkish_prior_delta():
    E = ORIG["model.decoder.embed_tokens.weight"].float().cpu()
    V, D = E.shape
    text = (ROOT / "tr_corpus.txt").read_text(encoding="utf-8").splitlines()
    cnt = Counter()
    for line in text:
        for variant in (line.strip(), line.strip().lower()):
            cnt.update(tok.encode(" " + variant, add_special_tokens=False))
    n_text = tok.eos_token_id
    N = sum(cnt.values())
    o = torch.zeros(V)
    for t in range(n_text):
        o[t] = torch.log(torch.tensor((cnt.get(t, 0) + 0.5) / (N + 0.5 * n_text)))
    o[:n_text] -= o[:n_text].mean()

    # Keep special tokens (EOS, timestamps, language tokens) anchored at 0 shift
    W = torch.ones(V)
    W[n_text:] = 25.0
    Es = E * W[:, None].sqrt()
    r = 1e-2 * (Es.T @ Es).trace() / D
    delta = torch.linalg.solve(Es.T @ Es + r * torch.eye(D), Es.T @ (o * W.sqrt()))

    # Strictly orthogonalize against E[eos_token_id] so EOS logit shift is 0.0000
    e_eos = E[tok.eos_token_id]
    delta = delta - (float(delta @ e_eos) / float(e_eos @ e_eos)) * e_eos

    shift = E[:n_text] @ delta
    delta = delta / shift.std()
    return delta


_cache = {}


def edit_A(lam: float) -> int:
    if "A" not in _cache:
        _cache["A"] = turkish_prior_delta()
    b = model.model.decoder.layer_norm.bias
    b.data.add_((lam * _cache["A"]).to(b.device, b.dtype))
    return 512


# ---------------- Edit B: Contrastive sharpening of <|tr|> language token (512 params) ----------------
def edit_B(beta: float) -> int:
    if "B_langs" not in _cache:
        tr_id = VOCAB["<|tr|>"]
        other_ids = [
            i for k, i in VOCAB.items()
            if k.startswith("<|") and k.endswith("|>") and len(k) in (6, 7) and k[2:-2].isalpha() and i != tr_id
        ]
        _cache["B_langs"] = (tr_id, other_ids)
    tr_id, other_ids = _cache["B_langs"]
    emb = model.model.decoder.embed_tokens.weight
    mu = emb.data[other_ids].mean(0)
    emb.data[tr_id].add_(beta * (emb.data[tr_id] - mu))
    return 512


# ---------------- Edit C: Encoder LayerNorm gain (1024 params) -> cures repetition loops ----------------
def edit_C(alpha: float) -> int:
    model.model.encoder.layer_norm.weight.data.mul_(1.0 + alpha)
    model.model.encoder.layer_norm.bias.data.mul_(1.0 + alpha)
    return 1024


# ---------------- Edit D: Decoder Layer 5 Cross-Attention boost (262k params) ----------------
def edit_D(alpha: float) -> int:
    w = model.model.decoder.layers[5].encoder_attn.out_proj.weight
    w.data.mul_(1.0 + alpha)
    return w.numel()


# ---------------- Edit E: Nullspace calibration of bound Turkish suffixes (fixes word splitting) ----------------
BOUND_SUFFIXES = [
    # Copula / case / plural /derivational suffixes that cannot stand alone as space-prefixed words in Turkish
    "dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür",
    "daki", "deki", "taki", "teki", "ndaki", "ndeki",
    "lar", "ler", "ları", "leri", "ların", "lerin", "lara", "lere", "lardan", "lerden",
    "dan", "den", "tan", "ten", "ndan", "nden",
    "nın", "nin", "nun", "nün", "ın", "in", "un", "ün",
    "lık", "lik", "luk", "lük", "lığı", "liği", "luğu", "lüğü",
    "mış", "miş", "muş", "müş", "iyor", "ıyor", "uyor", "üyor",
    "acak", "ecek", "acağı", "eceği", "cık", "cik", "cuk", "cük",
    "ca", "ce", "ça", "çe", "yla", "yle", "la", "le", "na", "ne",
    "nda", "nde",
]


def _build_suffix_pairs():
    sp_ids, ns_ids = [], []
    for sfx in BOUND_SUFFIXES:
        # Look up exact token IDs for " <sfx>" (spurious space) and "<sfx>" (bound suffix)
        ids_sp = tok.encode(" " + sfx, add_special_tokens=False)
        ids_ns = tok.encode(sfx, add_special_tokens=False)
        if len(ids_sp) == 1:
            sp_ids.append(ids_sp[0])
        if len(ids_ns) == 1:
            ns_ids.append(ids_ns[0])
    return sorted(set(sp_ids)), sorted(set(ns_ids))


def edit_E(delta_logit: float) -> int:
    if "E" not in _cache:
        _cache["E"] = _build_suffix_pairs()
    sp_ids, ns_ids = _cache["E"]
    u, bu = nullspace_unit_vec()
    step = (u / bu).to(model.device, model.dtype)
    emb = model.model.decoder.embed_tokens.weight.data
    # Suppress spurious space-prefixed bound suffix tokens by -delta_logit, boost bound no-space tokens by +0.5*delta_logit
    for i in sp_ids:
        emb[i].add_(-delta_logit * step)
    for i in ns_ids:
        emb[i].add_(+0.5 * delta_logit * step)
    return (len(sp_ids) + len(ns_ids)) * 512


# ---------------- Edit F: Nullspace suppression of non-Turkish Latin (q, w, x) & foreign tokens ----------------
def _build_foreign_tokens():
    bad_ids = []
    bad_chars = set("qwxQWXäßłńřãéèêñåøæÄßŁŃŘÃÉÈÊÑÅØÆ")
    n_text = tok.eos_token_id
    for t_str, idx in VOCAB.items():
        if idx >= n_text:
            continue
        s = tok.convert_tokens_to_string([t_str])
        if any(c in bad_chars for c in s):
            bad_ids.append(idx)
    return sorted(set(bad_ids))


def edit_F(delta_logit: float) -> int:
    if "F" not in _cache:
        _cache["F"] = _build_foreign_tokens()
    bad_ids = _cache["F"]
    u, bu = nullspace_unit_vec()
    step = (u / bu).to(model.device, model.dtype)
    emb = model.model.decoder.embed_tokens.weight.data
    emb[bad_ids] -= delta_logit * step[None, :]
    return len(bad_ids) * 512


# ---------------- Edit G: Surgical Attention-Head Loop Repair (L4.self.H4, L5.self.H1, L3.enc.H0) ----------------
def edit_G(gamma: float) -> int:
    # Dampen the two late-layer autoregressive copy heads (L4.self.H4, L5.self.H1) and boost L3.enc.H0 alignment head
    w4 = model.model.decoder.layers[4].self_attn.out_proj.weight.data
    w5 = model.model.decoder.layers[5].self_attn.out_proj.weight.data
    e3 = model.model.decoder.layers[3].encoder_attn.out_proj.weight.data
    w4[:, 4 * 64:5 * 64].mul_(1.0 - gamma)
    w5[:, 1 * 64:2 * 64].mul_(1.0 - gamma)
    e3[:, 0 * 64:1 * 64].mul_(1.0 + gamma)
    return 3 * 512 * 64


EDITS = {"A": edit_A, "B": edit_B, "C": edit_C, "D": edit_D, "E": edit_E, "F": edit_F, "G": edit_G}


def apply_spec(spec: dict) -> int:
    restore()
    params_edited = 0
    for k, v in spec.items():
        if abs(v) > 0:
            params_edited += EDITS[k](v)
    return params_edited


def run_eval(spec: dict, samples, tag: str, save: Path | None = None, **eval_kw):
    n_edited = apply_spec(spec)
    r = evaluate(proc, model, samples, tag, save=save, **eval_kw)
    r["spec"] = spec
    r["params_edited"] = n_edited
    r["pct_edited"] = round(100.0 * n_edited / 72593920, 4)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(r) + "\n")
    print(f"{fmt(r)}  edited={n_edited} ({r['pct_edited']:.4f}%)", flush=True)
    return r


if __name__ == "__main__":
    dev = load_split("dev")
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if mode == "sweep":
        done = set()
        if LOG.exists():
            for line in LOG.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done.add(json.loads(line)["tag"])

        def maybe(spec, tag):
            if tag not in done:
                run_eval(spec, dev, tag, bf16=True, max_new_tokens=115)

        maybe({}, "dev/base")
        for lam in (0.05, 0.08, 0.12, 0.15, 0.2, 0.3):
            maybe({"A": lam}, f"dev/A={lam}")
        for beta in (0.05, 0.1, 0.2, 0.35):
            maybe({"B": beta}, f"dev/B={beta}")
        for alpha in (0.05, 0.1, 0.2, 0.3):
            maybe({"C": alpha}, f"dev/C={alpha}")
        for alpha in (0.1, 0.2, 0.3):
            maybe({"D": alpha}, f"dev/D={alpha}")
        for d_e in (0.2, 0.4, 0.8):
            maybe({"E": d_e}, f"dev/E={d_e}")
        for d_f in (0.3, 0.7, 1.5):
            maybe({"F": d_f}, f"dev/F={d_f}")
    elif mode == "probe":
        spec = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in sys.argv[2:]}
        run_eval(spec, dev, f"dev/{spec}")
    elif mode == "final":
        spec = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in sys.argv[2:]}
        r_dev = run_eval(spec, dev, f"final/dev {spec}", save=OUT / "wsp_dev.json")
        test = load_split("test")
        r_test = run_eval(spec, test, f"final/test {spec}", save=OUT / "wsp_test.json")
        apply_spec(spec)
        out_weights = OUT / "whisper-base-tr-wsp.safetensors"
        save_file({k: v.contiguous() for k, v in model.state_dict().items()
                   if k != "proj_out.weight"}, str(out_weights))
        print("Saved edited weights ->", out_weights, flush=True)
