# Turkish STT: Baseline vs. Weight-Space Programming (WSP) vs. LoRA

A 3-way comparison on **`openai/whisper-base`** (72.6M parameters) for Turkish Speech-to-Text (`google/fleurs` `tr_tr`):
1. **Untouched Baseline** — `openai/whisper-base` out of the box.
2. **Weight-Space Programming (WSP)** — training-free, gradient-free surgical weight edits (`0.0628%` of parameters) designed by inspecting attention heads and LayerNorm nullspace geometry.
3. **LoRA Fine-Tuning** — gradient-based low-rank adaptation trained on FLEURS Turkish `train` on Google Colab (A100 GPU) and hot-swapped locally (`~4.2 MB` adapter).

---

## 3-Way Benchmark Results (`fp32`, `num_beams=1`, Turkish-aware normalizer)

| Method | Training Data | Gradient Steps | Params Changed / Added | Dev WER (338) | Dev CER | Dev Loops | Test WER (743) | Test CER | Test Loops | Test Clips Improved / Worse |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Untouched Baseline** | None | 0 | 0 (`0.00%`) | `33.82%` | `10.75%` | `1` | `30.79%` | `8.35%` | `0` | — |
| **WSP (`G=0.12, N=1.0, E=1.0`)** | None (`0` clips) | 0 | `45,568` (`0.06%`) | `30.24%` | `8.04%` | `0` | `30.78%` | `8.31%` | `0` | `54` / `51` |
| **LoRA Fine-Tune (Colab A100)** | FLEURS `train` | ~600 steps | `1,179,648` (`1.62%`) | `28.50%` | `7.25%` | `0` | **`27.42%`** | **`7.34%`** | `1` | **`344` / `225`** |

### How the WSP Edit (`0.0628%`) Works
- **`Edit G` (`32,768` params):** Scales `L2.encoder_attn.out_proj` head 6 by `-12%` (isolated via 96-head leave-one-out attribution) to stop repetition loops without hurting clean self-attention.
- **`Edit N` + `Edit E` (`12,800` params):** Shifts 25 specific Turkish number/unit and bound-suffix rows in `embed_tokens.weight` along the exact `decoder.layer_norm` nullspace direction ($u = \gamma^{-1} / \|\gamma^{-1}\|_2$), adjusting token logits cleanly without touching encoder LayerNorm or short-clip silence padding.

---

## Advantages & Disadvantages

- **Untouched Baseline (`openai/whisper-base`)**
  - **Pros:** Zero setup; balanced multilingual prior; no extra files.
  - **Cons:** ~30.8% Turkish WER; prone to repetition loops and weak Turkish suffix/number formatting.
- **Weight-Space Programming (WSP)**
  - **Pros:** **Zero training data, zero gradients, ~1 second to apply**; eliminates repetition loops (`1 -> 0`) and drops `dev` WER by `-3.58%` (`33.82% -> 30.24%`) with minimal collateral regressions (`54` improved vs `51` worse on `test`).
  - **Cons:** Cannot teach new Turkish vocabulary or acoustics it never learned; nullspace/head edits mostly generalize to known failure modes (`-0.01%` net on `test`).
- **LoRA Fine-Tuning (Colab A100)**
  - **Pros:** Directly learns Turkish acoustics, suffixes, and spelling; biggest real-world gain (`30.79% -> 27.42%` test WER, `344` clips improved); tiny hot-swappable `~4.2 MB` adapter.
  - **Cons:** Needs ~1.8 GB labeled Turkish audio and GPU training time (~10–15 min on Colab A100); can still occasionally loop (`1` loop on `test`).

---

## Side-by-Side Test Examples

1. **Rescuing a collapsed baseline (`Clip #443`)**
   - **Ref:** `latin dil ailesinden bir dil biliyorsanız portekizce öğrenmenin daha kolay olacağını söylemeye bile gerek yok`
   - **Base (`100.0%` WER):** `4`
   - **WSP (`40.0%` WER):** `nettin değil ailesinden bir değil biliyorsanız portakizcü öğrenmenin daha kolay olacağını söylemeye bile gerekiyor`
   - **LoRA (`46.7%` WER):** `latın dill ailesinden bir değil biliyorsunuz portakizcu öğrenmenin daha kolay olacağını söylemeye bile gerekiyor`

2. **LoRA fixing Turkish morphology & word boundaries (`Clip #98`)**
   - **Ref:** `bize treni arabayı ve daha birçok ulaşım aracını getirdi`
   - **Base (`44.4%` WER):** `bizet tireni arabayı ve daha bir çokulaşım aracını getirdi`
   - **WSP (`44.4%` WER):** `bizet tireni arabayı ve daha bir çokulaşım aracını getirdi`
   - **LoRA (`0.0%` WER):** `bize treni arabayı ve daha birçok ulaşım aracını getirdi`

3. **LoRA fixing proper nouns & loanwords (`Clip #561`)**
   - **Ref:** `1960 lar boyunca brzezinski john f kennedy nin danışmanı olarak ve sonra da lyndon b johnson yönetimnde çalıştı`
   - **Base (`61.1%` WER):** `1960 ler boyunca bir zezinski can f kennedin in danışmanı olarak ve sonra da lyndon bey i can sın yönetiminde çalıştı`
   - **WSP (`61.1%` WER):** `1960 ler boyunca bir zezinski can f kennedin in danışmanı olarak ve sonra da lyndon bey i can sın yönetiminde çalıştı`
   - **LoRA (`16.7%` WER):** `1960 lar boyunca bir zezinski john f kennedy nin danışmanı olarak ve sonra da lyndon b johnson yönetiminde çalıştı`

---

## Usage & Reproduction

### 1. Live Microphone Subtitles
Double-click [`run_subtitles.bat`](run_subtitles.bat) or run:
```powershell
.venv\Scripts\python.exe wsp\live_subtitles.py
```

### 2. Transcribe an Audio File (WSP)
```powershell
.venv\Scripts\python.exe wsp\transcribe_file.py path\to\audio.wav
```

### 3. Reproduce WSP & 3-Way Comparison
```powershell
cd wsp
..\.venv\Scripts\python.exe fetch_fast.py
..\.venv\Scripts\python.exe baseline.py
..\.venv\Scripts\python.exe edits.py final G=0.12 N=1.0 E=1.0
cd ..
.venv\Scripts\python.exe wsp\compare_3way.py --adapter wsp\out\whisper-base-tr-lora.zip
```
*(Train the LoRA adapter for free on Google Colab using [`wsp/colab_lora_train.ipynb`](wsp/colab_lora_train.ipynb).)*
