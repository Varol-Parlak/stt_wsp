# 3-Way Comparison: Baseline vs. Weight-Space Programming (WSP) vs. LoRA

Model: `openai/whisper-base` (72.6M params) evaluated identically (`fp32`, `num_beams=1`, Turkish-aware normalizer) on Google FLEURS Turkish (`tr_tr`).

## 1. Quantitative Results

| Method | Training Data | Gradient Steps | Params Changed / Added | Dev WER (338) | Dev CER | Dev Loops | Test WER (743) | Test CER | Test Loops | Test Clips Improved / Worse |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Untouched Baseline** | None | 0 | 0 (`0.00%`) | `33.82%` | `10.75%` | `1` | `30.79%` | `8.35%` | `0` | — |
| **WSP (Training-Free)** | None (0 clips) | 0 | 45,568 (`0.06%`) | `30.24%` | `8.04%` | `0` | `30.78%` | `8.31%` | `0` | `54` / `51` |
| **LoRA Fine-Tune (Colab T4)** | FLEURS `train` | ~600 steps | 1,179,648 (`1.62%`) | `28.50%` | `7.25%` | `0` | `27.42%` | `7.34%` | `1` | `344` / `225` |

## 2. Advantages & Disadvantages

| Approach | Advantages | Disadvantages |
| :--- | :--- | :--- |
| **Untouched Baseline** | Zero setup; balanced multilingual prior; no collateral regressions on clean words. | ~30.8% Turkish WER; susceptible to repetition loops (`bbbb...`) and weak cross-attention on suffixes. |
| **Weight-Space Programming (WSP)** | **Zero training data, zero gradients, ~1 second to apply**; eliminates repetition loops (`0 -> 0`) and recovers `+0.01%` test WER. | Cannot teach new Turkish vocabulary/morphology; manual weight scaling has a lower ceiling than gradient training. |
| **LoRA Fine-Tuning** | Directly teaches Turkish acoustics, suffixes, and orthography; largest WER reduction (`30.79% -> 27.42%`); hot-swappable ~4 MB adapter. | Requires ~1.8 GB labeled Turkish audio and GPU training time (~10–15 min on Colab T4); can still occasionally loop (`1` on test). |

## 3. Side-by-Side Test Transcripts

**Clip #439** (WER: Base `100.0%` | WSP `100.0%` | LoRA `31.6%`)
- **Ref**: `ana sahnelerde müzik bitmiş olsa bile festivalin gece geç saatlere kadar müzik çalmaya devam edecek olan kısımları olabileceğini unutmayın`
- **Base**: `anlaşılmıyorum`
- **WSP**: `anlaşılmıyorum`
- **LoRA**: `anlı sahnilerde müzik bilmiş olsa bile festivalin gece gecesi kadar müzik çalmayı devam edecek olan kısımları olabileceğini unutmayın`

**Clip #558** (WER: Base `66.7%` | WSP `66.7%` | LoRA `11.1%`)
- **Ref**: `bize treni arabayı ve daha birçok ulaşım aracını getirdi`
- **Base**: `bizet tüleni ara boyu ve daha bir çok ulaşım aracını getirdi`
- **WSP**: `bizet tüleni ara boyu ve daha bir çok ulaşım aracını getirdi`
- **LoRA**: `bizet treni arabayı ve daha birçok ulaşım aracını getirdi`

**Clip #443** (WER: Base `100.0%` | WSP `40.0%` | LoRA `46.7%`)
- **Ref**: `latin dil ailesinden bir dil biliyorsanız portekizce öğrenmenin daha kolay olacağını söylemeye bile gerek yok`
- **Base**: `4`
- **WSP**: `nettin değil ailesinden bir değil biliyorsanız portakizcü öğrenmenin daha kolay olacağını söylemeye bile gerekiyor`
- **LoRA**: `latın dill ailesinden bir değil biliyorsunuz portakizcu öğrenmenin daha kolay olacağını söylemeye bile gerekiyor`

**Clip #561** (WER: Base `61.1%` | WSP `61.1%` | LoRA `16.7%`)
- **Ref**: `1960 lar boyunca brzezinski john f kennedy nin danışmanı olarak ve sonra da lyndon b johnson yönetimnde çalıştı`
- **Base**: `1960 ler boyunca bir zezinski can f kennedin in danışmanı olarak ve sonra da lyndon bey i can sın yönetiminde çalıştı`
- **WSP**: `1960 ler boyunca bir zezinski can f kennedin in danışmanı olarak ve sonra da lyndon bey i can sın yönetiminde çalıştı`
- **LoRA**: `1960 lar boyunca bir zezinski john f kennedy nin danışmanı olarak ve sonra da lyndon b johnson yönetiminde çalıştı`

**Clip #98** (WER: Base `44.4%` | WSP `44.4%` | LoRA `0.0%`)
- **Ref**: `bize treni arabayı ve daha birçok ulaşım aracını getirdi`
- **Base**: `bizet tireni arabayı ve daha bir çokulaşım aracını getirdi`
- **WSP**: `bizet tireni arabayı ve daha bir çokulaşım aracını getirdi`
- **LoRA**: `bize treni arabayı ve daha birçok ulaşım aracını getirdi`

**Clip #630** (WER: Base `58.3%` | WSP `58.3%` | LoRA `16.7%`)
- **Ref**: `çoğu modern araştırma teleskobu uygun atmosfer koşullarına sahip ücra bölgelerdeki devasa tesislerdir`
- **Base**: `çuğu modern araştırma tilesi kovu uygun atmosfer kuşurlarına sahip ücrab ölgelerdeki devasa teslislidir`
- **WSP**: `çuğu modern araştırma tilesi kovu uygun atmosfer kuşurlarına sahip ücrab ölgelerdeki devasa teslislidir`
- **LoRA**: `çoğu modern araştırma trescobu uygun atmosfer koşurlarına sahip ücra bölgelerdeki devasa tesislerdir`
