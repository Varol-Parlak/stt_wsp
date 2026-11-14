"""Network-friendly fetch: parallel range download for model files + per-clip audio via HF datasets-server API.
(The HF CDN gives ~100 KB/s per connection on this network, so we open many connections.)"""
import os
import io
import json
import time
import concurrent.futures as cf
from pathlib import Path

import requests
from common import DATA, MODEL_ID

HF = "https://huggingface.co"
S = requests.Session()


def get(url, **kw):
    for i in range(30):
        try:
            r = S.get(url, timeout=60, **kw)
            if r.status_code in (200, 206):
                return r
            print("  http", r.status_code, url[:90], flush=True)
        except Exception as e:
            print("  retry", i, repr(e)[:120], flush=True)
        time.sleep(3)
    raise RuntimeError(url)


def parallel_download(url, dst: Path, n_conn=16, chunk=4 * 2**20):
    if dst.exists():
        return
    head = S.head(url, allow_redirects=True, timeout=60)
    size = int(head.headers["Content-Length"])
    final = head.url
    ranges = [(s, min(s + chunk, size) - 1) for s in range(0, size, chunk)]
    parts = {}
    t0 = time.time()

    def job(rg):
        return rg[0], get(final, headers={"Range": f"bytes={rg[0]}-{rg[1]}"}).content

    with cf.ThreadPoolExecutor(n_conn) as ex:
        for k, (start, data) in enumerate(ex.map(job, ranges)):
            parts[start] = data
            done = sum(map(len, parts.values()))
            if k % 5 == 0:
                print(f"  {dst.name}: {done/2**20:.0f}/{size/2**20:.0f} MB  {done/2**20/(time.time()-t0):.2f} MB/s", flush=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        for s in sorted(parts):
            f.write(parts[s])
    assert dst.stat().st_size == size


def fetch_model():
    d = DATA / "models" / MODEL_ID.replace("/", "__")
    files = ["config.json", "generation_config.json", "preprocessor_config.json", "tokenizer_config.json",
             "tokenizer.json", "vocab.json", "merges.txt", "normalizer.json", "added_tokens.json",
             "special_tokens_map.json", "model.safetensors"]
    for f in files:
        url = f"{HF}/{MODEL_ID}/resolve/main/{f}"
        if f.endswith(".safetensors"):
            parallel_download(url, d / f)
        elif not (d / f).exists():
            r = S.get(url, timeout=60)
            if r.status_code == 200:
                d.mkdir(parents=True, exist_ok=True)
                (d / f).write_bytes(r.content)
    print("model ->", d, flush=True)


def fetch_split(split, dst, n):
    out = DATA / f"fleurs_tr_{dst}"
    out.mkdir(parents=True, exist_ok=True)
    meta_p = out / "meta.json"
    rows = []
    for off in range(0, n, 100):
        api = (f"https://datasets-server.huggingface.co/rows?dataset=google/fleurs&config=tr_tr"
               f"&split={split}&offset={off}&length={min(100, n - off)}")
        rows += [r["row"] for r in get(api).json()["rows"]]

    def dl(r):
        p = out / f"{r['id']}.wav"
        if not p.exists():
            src = r["audio"][0]["src"] if isinstance(r["audio"], list) else r["audio"]["src"]
            p.write_bytes(get(src).content)
        return {"id": r["id"], "file": p.name, "text": r["raw_transcription"]}

    with cf.ThreadPoolExecutor(16) as ex:
        meta = list(ex.map(dl, rows))
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{dst}: {len(meta)} clips -> {out}", flush=True)


if __name__ == "__main__":
    fetch_split("validation", "dev", 150)
    fetch_split("test", "test", 200)
    fetch_model()
