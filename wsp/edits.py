"""Weight-Space Programming (WSP) interventions for Turkish Whisper (openai/whisper-base).

Each edit is a closed-form, training-free, gradient-free surgical modification to specific weight tensors:
  - Edit A (512 params):    Turkish lexical prior projected into `decoder.layer_norm.bias` (orthogonal to EOS).
  - Edit B (512 params):    Contrastive sharpening of the `<|tr|>` language token in `embed_tokens.weight`.
  - Edit C (1024 params):   Encoder `layer_norm` gain (legacy; amplifies 30s silence padding on short clips).
  - Edit D (262k params):   Decoder Layer-5 cross-attention `out_proj` gain.
  - Edit E (7,680 params):  Pure Nullspace Suppression along u = 1/gamma for Turkish bound clitics/suffixes & morphology (15 tokens).
  - Edit F (~80k params):   Nullspace logit suppression along u = 1/gamma for non-Turkish alphabet tokens (q, w, x, foreign).
  - Edit G (32,768 params): Surgical Cross-Attention Head De-Looping on `L2.enco.H6` (-12%), identified via 96-head leave-one-out attribution.
  - Edit N (5,120 params):  Pure Nullspace Suppression along u = 1/gamma for spoken Turkish number words & units vs Arabic digits (10 tokens).

Best Surgical WSP Configuration (`G=0.12 N=1.0 E=1.0`, 45,568 params = 0.0628% of model):
  - Zero encoder LayerNorm scaling (C=0), preserving 100% silence-padding stability on short 1-3s utterances.
  - Pure nullspace suppression on tied `embed_tokens.weight`: zero effect on clips not emitting target tokens, and suppressed tokens are not fed back into Layer 0.

Usage:
  python edits.py sweep                       # sweep individual edits & combinations on dev (338 clips)
  python edits.py probe G=0.12 N=1.0 E=1.0    # evaluate specific edit spec on dev
  python edits.py final G=0.12 N=1.0 E=1.0    # apply best spec, evaluate on dev + held-out test (743 clips), save weights
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
    "model.decoder.layers.2.encoder_attn.out_proj.weight",
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
# Thus shifting a token embedding E_v by (-delta / (beta . u)) * u shifts token v's output logit by
# EXACTLY -delta at every step without perturbing the acoustic/contextual dot product (gamma * h_hat) . E_v.
# By using PURE SUPPRESSION (-delta * step, delta > 0) on over-emitted tokens v, any clip that does not emit v
# is 100% unaffected, and once v is suppressed below the runner-up token w, v is never emitted into Layer 0.
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


# ---------------- Edit C: Encoder LayerNorm gain (1024 params) [Legacy] ----------------
def edit_C(alpha: float) -> int:
    model.model.encoder.layer_norm.weight.data.mul_(1.0 + alpha)
    model.model.encoder.layer_norm.bias.data.mul_(1.0 + alpha)
    return 1024


# ---------------- Edit D: Decoder Layer 5 Cross-Attention boost (262k params) ----------------
def edit_D(alpha: float) -> int:
    w = model.model.decoder.layers[5].encoder_attn.out_proj.weight
    w.data.mul_(1.0 + alpha)
    return w.numel()


# ---------------- Edit N: Spoken Turkish Number Words & Units Nullspace Calibration (5,120 params) ----------------
# Suppresses over-emitted Arabic digit / symbol tokens along u = 1/gamma so spoken Turkish number words
# (iki, üç, dört, beş, altı, sekiz, on, bin, yüzde, m) win when pronounced in Turkish audio.
NUMBER_UNIT_SUPPRESS = {
    568: 1.8,    # ' 2' -> iki
    805: 2.5,    # ' 3' -> üç
    1017: 3.5,   # ' 4' -> dört
    1025: 2.5,   # ' 5' -> beş
    1386: 2.5,   # ' 6' -> altı / altın
    1649: 1.5,   # ' 8' -> sekiz
    1266: 2.0,   # ' 10' -> on
    9714: 1.2,   # ' 1000' -> bin
    14189: 1.5,  # ' %' -> yüzde
    42431: 2.0,  # ' metre' -> m (spelled abbreviation)
}


def edit_N(scale: float) -> int:
    u, bu = nullspace_unit_vec()
    step = (u / bu).to(model.device, model.dtype)
    emb = model.model.decoder.embed_tokens.weight.data
    for tid, delta in NUMBER_UNIT_SUPPRESS.items():
        emb[tid].add_(-(scale * delta) * step)
    return len(NUMBER_UNIT_SUPPRESS) * 512


# ---------------- Edit E: Turkish Bound Clitics, Suffixes & Morphology Nullspace Calibration (7,680 params) ----------------
# Pure nullspace suppression along u = 1/gamma for space-prefixed clitics/suffixes that cause spurious word splitting
# (' ile' -> -yla/-yle, ' ki' -> -ki, ' dir' -> -dir, ' hang' -> herhangi, ' kaç' -> birkaç) and high-frequency
# Turkish vowel-harmony / morphological suffix confusions.
MORPH_SUFFIX_SUPPRESS = {
    15465: 1.5,  # ' ile' -> bound instrumental enclitic -yla/-yle
    6315: 2.8,   # ' ki' -> bound relativizer suffix -ki (müzedeki, ülkedeki)
    4746: 2.0,   # ' dir' -> bound copula suffix -dir (vergilendirildi)
    3967: 1.5,   # ' hang' -> compound herhangi
    23916: 0.8,  # ' kaç' -> compound birkaç
    3812: 0.8,   # 'ini' -> accusative/dative -ini vs -ine/-inde
    374: 1.2,    # 'ur' -> vowel harmony suffix
    5575: 2.2,   # 'anda' -> locative suffix confusion
    21006: 0.8,  # 'ının' -> double-genitive haplology -ının -> -ın
    2790: 1.5,   # 'eral' -> yeral -> yerel vowel harmony
    34: 1.5,     # 'C' -> mid-word uppercase C
    4934: 1.0,   # 'üz' -> vowel harmony suffix
    3004: 2.0,   # 'ged' -> stem confusion
    22569: 2.0,  # 'ımı' -> possessive suffix harmony
    38837: 1.6,  # 'brahim' -> İbrani vs İbrahim disambiguation
}


def edit_E(scale: float) -> int:
    u, bu = nullspace_unit_vec()
    step = (u / bu).to(model.device, model.dtype)
    emb = model.model.decoder.embed_tokens.weight.data
    for tid, delta in MORPH_SUFFIX_SUPPRESS.items():
        emb[tid].add_(-(scale * delta) * step)
    return len(MORPH_SUFFIX_SUPPRESS) * 512


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


# ---------------- Edit G: Surgical Cross-Attention Head De-Looping on L2.enco.H6 (32,768 params) ----------------
def edit_G(gamma: float) -> int:
    # Dampen Decoder Layer 2 Cross-Attention Head 6 (L2.enco.H6), identified via 96-head leave-one-out
    # logit attribution as the specific head driving repetition loops, with minimal clean-clip perturbation.
    e2 = model.model.decoder.layers[2].encoder_attn.out_proj.weight.data
    e2[:, 6 * 64:7 * 64].mul_(1.0 - gamma)
    return 512 * 64


EDITS = {
    "A": edit_A,
    "B": edit_B,
    "C": edit_C,
    "D": edit_D,
    "E": edit_E,
    "F": edit_F,
    "G": edit_G,
    "N": edit_N,
}


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
                run_eval(spec, dev, tag)

        maybe({}, "dev/base")
        maybe({"G": 0.12}, "dev/G=0.12_L2H6")
        maybe({"N": 1.0}, "dev/N=1.0_numbers")
        maybe({"E": 1.0}, "dev/E=1.0_morph")
        maybe({"N": 1.0, "E": 1.0}, "dev/N=1.0+E=1.0")
        maybe({"G": 0.12, "N": 1.0, "E": 1.0}, "dev/G=0.12+N=1.0+E=1.0")
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
