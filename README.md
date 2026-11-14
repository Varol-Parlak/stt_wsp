# Weight-Space Programming (WSP) for Turkish Speech-to-Text (`whisper-base`)

Direct, training-free, gradient-free **Weight-Space Programming (WSP)** on `openai/whisper-base` (72.6M parameters) to improve **Turkish Speech-to-Text (STT)**—inspired by *"Sparks of Weight-Space Programming"* (Takehi, Teoh, Zhang, Khattab, MIT CSAIL 2026).

Without running a single step of backpropagation, gradient descent, or fine-tuning, we inspect the model's weight matrices and LayerNorm geometry, diagnose its structural failure modes on Turkish speech (autoregressive repetition loops, weak encoder-to-decoder cross-attention gain, English/foreign vocabulary leakage, and agglutinative suffix splitting), and directly edit **0.1375% of the weights (99,840 parameters)**.

---

## Results Summary (Google FLEURS Turkish `tr_tr`)

Evaluated in full `float32` (`num_beams=1`, `max_new_tokens=200`) across all **338 validation (`dev`) clips** and **743 held-out `test` clips**:

| Model | Split | Clips | WER ↓ | CER ↓ | Word Loops | Edited Params |
|---|---|---:|---:|---:|---:|---:|
| `openai/whisper-base` (Untouched Baseline) | `dev` | 338 | 33.82% | 10.75% | 1 | 0 (0.0000%) |
| **`whisper-base-tr-wsp` (`C=0.05, G=0.15, B=0.03`)** | **`dev`** | **338** | **31.83%** (**-1.99%**) | **8.31%** (**-2.44% / -22.7% rel.**) | **0** | **99,840 (0.1375%)** |
| `openai/whisper-base` (Untouched Baseline) | `test` (held-out) | 743 | 30.79% | 8.35% | 0 | 0 (0.0000%) |
| **`whisper-base-tr-wsp` (`C=0.05, G=0.15, B=0.03`)** | **`test` (held-out)** | **743** | **30.12%** (**-0.67%**) | **7.71%** (**-0.64% / -7.7% rel.**) | **0** | **99,840 (0.1375%)** |

* **58 clips improved** on `dev` and **151 clips improved** on the held-out `test` set.

---

## Mechanistic Diagnosis & Weight-Space Interventions

### 1. Acoustic Cross-Attention Gain (`Edit C` — 1,024 parameters, `0.0014%`)
In `WhisperDecoderLayer`, `encoder_attn_layer_norm` only normalizes the decoder Query $Q$, while the Keys $K$ and Values $V$ in every cross-attention layer come directly from `model.encoder.layer_norm`. Inspecting the weight statistics ([`wsp/inspect_weights.py`](wsp/inspect_weights.py)) revealed that `model.encoder.layer_norm.weight` has mean `+1.351`, whereas `model.decoder.layer_norm.weight` has mean `+9.088`. When the English-dominated decoder language model is uncertain on a Turkish word, autoregressive `self_attn` overpowers acoustic `encoder_attn`, causing hallucinations, early `<|endoftext|>`, or repetition loops.
* **Intervention**: Scale `model.encoder.layer_norm.weight` and `model.encoder.layer_norm.bias` by $1 + \alpha$ ($\alpha = 0.05$).

### 2. Surgical Attention-Head Loop Repair (`Edit G` — 98,304 parameters, `0.1354%`)
Scanning all 48 causal `self_attn` heads and 48 `encoder_attn` heads across the 6 decoder layers isolated the exact attention heads responsible for repetitive decoding loops:
* **Repetition / Self-Copy Heads**: `decoder.layers[4].self_attn` **Head 4** and `decoder.layers[5].self_attn` **Head 1** copy the immediately preceding token into the residual stream.
* **Acoustic Alignment Head**: `decoder.layers[3].encoder_attn` **Head 0** advances the cross-attention alignment pointer along the audio frames.
* **Intervention**: Dampen `L4.self_attn.H4` and `L5.self_attn.H1` by $1 - \gamma$ and boost `L3.enc_attn.H0` by $1 + \gamma$ ($\gamma = 0.15$) in `out_proj.weight[:, h*64:(h+1)*64]`.

### 3. Contrastive `<|tr|>` Language-Token Steering (`Edit B` — 512 parameters, `0.0007%`)
In `embed_tokens.weight`, the Turkish conditioning token `<|tr|>` (`id = 50268`) sits inside a dense multilingual cluster (cosine similarity `0.683` with `<|en|>`, `0.848` with `<|az|>`). Because special tokens are suppressed in `proj_out` during generation, editing row `50268` of `embed_tokens.weight` affects **only** the prefix conditioning vector injected at position 1 of the decoder:
$$E_{\text{tr}} \leftarrow E_{\text{tr}} + \beta \left(E_{\text{tr}} - \mu_{\text{other langs}}\right), \quad \beta = 0.03$$

### 4. Exact LayerNorm-Nullspace Logit Calibration (`Edits A, E, F`)
Because `proj_out.weight` is tied to `model.decoder.embed_tokens.weight` ($E$), output logits at step $t$ are:
$$z_{t, v} = (\gamma \odot \text{LN}_0(h_t) + \beta) \cdot E_v$$
Since $\text{LN}_0(h_t)$ subtracts the mean across the 512 channels, $\sum_{i=1}^{512} \text{LN}_0(h_t)_i \equiv 0$ for **all** $h_t \in \mathbb{R}^{512}$. Consequently, the vector $u = \gamma^{-1} / \|\gamma^{-1}\|_2$ lies in the **exact nullspace** of $\gamma \odot \text{LN}_0(h_t)$ ($\max |(\gamma \odot \text{LN}_0(h_t)) \cdot u| = 4.5 \times 10^{-6}$), while $\beta \cdot u = -3.1194 \neq 0$. Adding $\frac{\delta_v}{\beta \cdot u} u$ to any token row $E_v$ shifts token $v$'s output logit by **exactly $+\delta_v$** without distorting its acoustic/contextual dot product $(\gamma \odot \text{LN}_0(h_t)) \cdot E_v$.

---

## Qualitative Examples (Before vs. After WSP)

**1. Curing Word & Character Repetition Loops (`dev` `0171_1582` & `test` `0212_1832`)**
* **Reference**: `futbol basketbol voleybol sutopu eskrim ragbi bisiklete binme buz hokeyi patenli hokey ve f1 motorlu araç yarışları popüler sporlardandır`
* **Untouched `whisper-base`**: `fotovasket value bolsutopu xgrim rag mmsk bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb`
* **WSP (`C=0.05, G=0.15, B=0.03`)**: `football basketball volleyballsutopu x cream ragnb mmsk te bilme buz okeyi batalin okey ve f1 motorlu araç yarışları popüler sporlardandır`

**2. Preventing Premature Hallucinated EOS (`test` `0439_1771`)**
* **Reference**: `ana sahnelerde müzik bitmiş olsa bile festivalin gece geç saatlere kadar müzik çalmaya devam edecek olan kısımları olabileceğini unutmayın`
* **Untouched `whisper-base`**: `anlaşılmıyorum`
* **WSP (`C=0.05, G=0.15, B=0.03`)**: `anasar nelerde müzik bilmiş olsa bile festivalin gece geç saatlere kadar müzik çalmayı devam edecek olan kısımları olabileceğini unutmayın`

**3. Eliminating English Lexical Leakage & Word Splitting (`dev` `0322_1585`, `0334_1528`, `test` `0188_1839`)**
* **Reference**: `zamanla yeni nüfus yeni çevresine uyum sağlamaya başladıkça diğer popülasyondan gittikçe daha farklı görünmeye başlarlar`
* **Untouched `whisper-base`**: `zamanla yenilip uymus alamaya başladıkça diğer population ile gidikçe daha farklı görünmeye başlarlar`
* **WSP (`C=0.05, G=0.15, B=0.03`)**: `zamanla yenilip us yeni çevresine uyum salamaya başladıkça diğer popülasyonla gittikçe daha farklı görünmeye başlarlar`
* **Reference**: `bazı festivaller küçük çocuklu aileler için özel kamp alanlarına sahiptir`
* **Untouched `whisper-base`**: `bazı festi varlar küçük çocuklu aileler için özel kam alanlarına sahiptir`
* **WSP (`C=0.05, G=0.15, B=0.03`)**: `bazı festivaller küçük çocuklu aileler için özel kamp alanlarına sahiptir`
* **Reference**: `kayak rotasını yürüyüş rotasının bir benzeri olarak düşünün`
* **Untouched `whisper-base`**: `kayak rotasını yürü iş rotasını bir benzeri olarak düşünün`
* **WSP (`C=0.05, G=0.15, B=0.03`)**: `kayak rotasını yürüyüş rotasının bir benzeri olarak düşünün`

---

## Repository Structure & Usage

* [`wsp/common.py`](wsp/common.py): Model loader, Turkish-aware text normalizer (`I -> ı`, `İ -> i`), memory-mapped feature/encoder caching, and WER/CER/loop evaluator.
* [`wsp/fetch_fast.py`](wsp/fetch_fast.py): Downloads `openai/whisper-base` and Google FLEURS Turkish (`tr_tr`) `dev` (338 clips) and `test` (743 clips).
* [`wsp/inspect_weights.py`](wsp/inspect_weights.py) & [`wsp/analyze.py`](wsp/analyze.py): Weight-space statistics, SVD effective rank, language embedding geometry, and error anatomy.
* [`wsp/edits.py`](wsp/edits.py): Implements all Weight-Space Programming edits (`A`–`G`), dev sweeps, and final model export (`wsp/out/whisper-base-tr-wsp.safetensors`).
* [`wsp/transcribe_file.py`](wsp/transcribe_file.py): CLI script to transcribe any Turkish audio file with the WSP-edited weights.
* [`wsp/out/`](wsp/out/): Full JSON evaluation outputs (`baseline_dev.json`, `baseline_test.json`, `wsp_dev.json`, `wsp_test.json`, and `edits_log.jsonl`).

### Reproduce
```bash
cd wsp
python fetch_fast.py
python baseline.py
python edits.py sweep
python edits.py final C=0.05 G=0.15 B=0.03
python transcribe_file.py path/to/turkish_audio.wav
```
