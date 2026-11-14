"""Fast path (good connection): download FLEURS tr_tr parquet + model directly, unpack to wav + meta.json."""
import os
import json
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import hf_hub_download, snapshot_download
import pyarrow.parquet as pq
from common import DATA, MODEL_ID

for split, dst in (("validation", "dev"), ("test", "test")):
    p = hf_hub_download("google/fleurs", f"tr_tr/{split}/0000.parquet", repo_type="dataset",
                        revision="refs/convert/parquet")
    out = DATA / f"fleurs_tr_{dst}"
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for r in pq.read_table(p, columns=["id", "audio", "raw_transcription"]).to_pylist():
        f = out / f"{r['id']}.wav"
        if not f.exists():
            f.write_bytes(r["audio"]["bytes"])
        meta.append({"id": r["id"], "file": f.name, "text": r["raw_transcription"]})
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(dst, len(meta), "clips", flush=True)

d = DATA / "models" / MODEL_ID.replace("/", "__")
snapshot_download(MODEL_ID, local_dir=d, allow_patterns=["*.json", "model.safetensors", "*.txt"])
print("model ->", d, flush=True)
