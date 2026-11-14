"""Step 1: baseline of the untouched model on Turkish + a look at what kind of errors it makes."""
import json
import sys
from collections import Counter
from common import ROOT, MODEL_ID, load_model, load_split, evaluate, fmt

N = int(sys.argv[1]) if len(sys.argv) > 1 else None
proc, model = load_model()
print("model:", MODEL_ID, f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

dev = load_split("dev", N)
test = load_split("test", N)
out = ROOT / "out"
r_dev = evaluate(proc, model, dev, "baseline/dev", save=out / "baseline_dev.json")
print(fmt(r_dev))
r_test = evaluate(proc, model, test, "baseline/test", save=out / "baseline_test.json", cache=False)
print(fmt(r_test))

# error anatomy on dev: which character substitutions dominate?
pairs = json.loads((out / "baseline_dev.json").read_text(encoding="utf-8"))["pairs"]
import jiwer
subs = Counter()
for p in pairs:
    if not p["hyp"]:
        continue
    al = jiwer.process_characters(p["ref"], p["hyp"])
    for chunk in al.alignments[0]:
        if chunk.type == "substitute":
            r = p["ref"][chunk.ref_start_idx:chunk.ref_end_idx]
            h = p["hyp"][chunk.hyp_start_idx:chunk.hyp_end_idx]
            if len(r) == len(h):
                for a, b in zip(r, h):
                    subs[(a, b)] += 1
print("\ntop char substitutions (ref->hyp):")
for (a, b), c in subs.most_common(20):
    print(f"  {a!r}->{b!r}: {c}")

print("\nworst dev examples:")
for p in sorted(pairs, key=lambda p: -jiwer.wer(p["ref"], p["hyp"] or "<empty>"))[:6]:
    print("  REF:", p["ref"][:150]); print("  HYP:", p["hyp"][:150]); print()
