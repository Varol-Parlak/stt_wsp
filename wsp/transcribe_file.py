"""Transcribe any Turkish audio file using the Weight-Space Programmed (WSP) Whisper model.

Usage:
  python transcribe_file.py path/to/audio.wav
"""
import sys
import torch
from pathlib import Path
from common import load_model, _read_audio, DEVICE

WEIGHTS = Path(__file__).resolve().parent / "out" / "whisper-base-tr-wsp.safetensors"


def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_file.py <audio_path>")
        sys.exit(1)
    audio_path = Path(sys.argv[1])
    state_path = str(WEIGHTS) if WEIGHTS.exists() else None
    proc, model = load_model(state_dict_path=state_path)
    wav = _read_audio(audio_path)
    feats = proc.feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE, model.dtype)
    ids = model.generate(feats, language="turkish", task="transcribe", num_beams=1, max_new_tokens=200)
    text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    print(text)


if __name__ == "__main__":
    main()
