"""Shared helpers for Weight-Space Programming on a small Whisper model (Turkish STT).

- load_model():     the target model whose weights we edit
- load_split():     FLEURS Turkish audio + reference transcripts (dev = edit against, test = held out)
- evaluate():       WER / CER plus failure diagnostics (repetition loops, wrong language, empty output)
"""
import os
import re
import json
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # xet backend stalls on this network

import numpy as np
import torch
import jiwer
import transformers
transformers.logging.set_verbosity_error()

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_ID = os.environ.get("WSP_MODEL", "openai/whisper-base")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- text normalization (Turkish-aware) ----------------
def tr_lower(s: str) -> str:
    # str.lower() maps I->i and İ->i̇, both wrong for Turkish
    return s.replace("I", "ı").replace("İ", "i").lower()


def norm(s: str) -> str:
    s = tr_lower(s)
    s = s.replace("â", "a").replace("î", "i").replace("û", "u")
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------- data ----------------
def _read_audio(path):
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:
        import torchaudio.functional as AF
        wav = AF.resample(torch.from_numpy(wav), sr, 16000).numpy()
    return wav


def load_split(split: str, n: int | None = None):
    """Returns list of dicts {id, path, text}; audio is read lazily per batch (saves RAM)."""
    d = DATA / f"fleurs_tr_{split}"
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return [{"id": r["id"], "path": d / r["file"], "text": r["text"]} for r in (meta[:n] if n else meta)]


# ---------------- model ----------------
def load_model(state_dict_path: str | None = None, dtype=torch.float32):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    local = DATA / "models" / MODEL_ID.replace("/", "__")
    src = str(local) if (local / "model.safetensors").exists() else MODEL_ID
    proc = WhisperProcessor.from_pretrained(src)
    model = WhisperForConditionalGeneration.from_pretrained(src, dtype=dtype)
    if state_dict_path:
        from safetensors.torch import load_file
        model.load_state_dict(load_file(state_dict_path), strict=False)
    return proc, model.to(DEVICE).eval()


_feat_cache = {}


def features(proc, samples, cache=True):
    """Log-mel features, cached on CPU in fp16 (they never change while we edit weights)."""
    out = []
    for s in samples:
        key = str(s["path"])
        if key not in _feat_cache:
            f = proc.feature_extractor(_read_audio(s["path"]), sampling_rate=16000,
                                       return_tensors="pt").input_features[0].half()
            if not cache:
                out.append(f); continue
            _feat_cache[key] = f
        out.append(_feat_cache[key])
    return torch.stack(out)


@torch.no_grad()
def transcribe(proc, model, samples, batch_size=24, cache=True, **gen_kw):
    kw = dict(language="turkish", task="transcribe", num_beams=1, max_new_tokens=200)
    kw.update(gen_kw)
    hyps = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        feats = features(proc, batch, cache=cache).to(DEVICE, model.dtype)
        ids = model.generate(feats, **kw)
        hyps += proc.batch_decode(ids, skip_special_tokens=True)
    return hyps


# ---------------- diagnostics ----------------
def is_loop(h: str) -> bool:
    w = h.split()
    if len(w) < 8:
        return False
    # any 1-4 gram repeated >= 4 times consecutively
    for k in range(1, 5):
        for i in range(len(w) - 4 * k + 1):
            g = w[i:i + k]
            if all(w[i + j * k:i + (j + 1) * k] == g for j in range(4)):
                return True
    return False


TR_CHARS = set("çğıöşü")


def evaluate(proc, model, samples, tag="", save: Path | None = None, **gen_kw):
    hyps_raw = transcribe(proc, model, samples, **gen_kw)
    refs = [norm(s["text"]) for s in samples]
    hyps = [norm(h) for h in hyps_raw]
    safe_h = [h if h else "<empty>" for h in hyps]
    ref_tr = sum(sum(c in TR_CHARS for c in r) for r in refs)
    hyp_tr = sum(sum(c in TR_CHARS for c in h) for h in hyps)
    res = {
        "tag": tag,
        "n": len(samples),
        "wer": jiwer.wer(refs, safe_h),
        "cer": jiwer.cer(refs, safe_h),
        "loops": sum(is_loop(h) for h in hyps),
        "empty": sum(not h for h in hyps),
        "tr_char_ratio": hyp_tr / max(ref_tr, 1),  # <1 => model drops Turkish letters
        "len_ratio": sum(len(h) for h in hyps) / max(sum(len(r) for r in refs), 1),
    }
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(json.dumps({"metrics": res, "pairs": [
            {"id": s["id"], "ref": r, "hyp": h} for s, r, h in zip(samples, refs, hyps)]},
            ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def fmt(r):
    return (f"[{r['tag']}] n={r['n']} WER={r['wer']:.4f} CER={r['cer']:.4f} loops={r['loops']} "
            f"empty={r['empty']} trChar={r['tr_char_ratio']:.3f} len={r['len_ratio']:.3f}")
