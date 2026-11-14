"""Live Microphone Turkish Subtitles using the Weight-Space Programmed (WSP) Whisper model.

Features:
- Captures 16 kHz mono audio from your microphone via `sounddevice`.
- Real-time streaming partial subtitles (~0.55s update interval while speaking)
  + utterance finalization on natural pauses (~0.65s silence).
- Always-on-top draggable desktop subtitle overlay (`tkinter`) + live terminal output.
- One-click live toggle between:
    * WSP Turkish (`C=0.05, G=0.15, B=0.03`)
    * Untouched Baseline (`openai/whisper-base`)
    * Side-by-Side Compare (runs both on each spoken utterance so you can see the difference live)

Usage:
  .venv\\Scripts\\python.exe wsp/live_subtitles.py
  .venv\\Scripts\\python.exe wsp/live_subtitles.py --no-gui
  .venv\\Scripts\\python.exe wsp/live_subtitles.py --device 1
"""
import sys
import time
import queue
import argparse
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import sounddevice as sd
import torch

from common import load_model, DEVICE

TOUCHED_KEYS = [
    "model.decoder.embed_tokens.weight",
    "model.encoder.layer_norm.weight",
    "model.encoder.layer_norm.bias",
    "model.decoder.layers.3.encoder_attn.out_proj.weight",
    "model.decoder.layers.4.self_attn.out_proj.weight",
    "model.decoder.layers.5.self_attn.out_proj.weight",
]


def apply_wsp_best(model, tok, c_alpha=0.05, g_gamma=0.15, b_beta=0.03):
    """Apply the best WSP Turkish configuration (C=0.05, G=0.15, B=0.03) in-place."""
    # Edit C: Encoder final LayerNorm gain (1,024 params)
    model.model.encoder.layer_norm.weight.data.mul_(1.0 + c_alpha)
    model.model.encoder.layer_norm.bias.data.mul_(1.0 + c_alpha)

    # Edit G: Dampen causal copy heads (L4.self.H4, L5.self.H1) & boost L3.enc.H0 alignment head (98,304 params)
    w4 = model.model.decoder.layers[4].self_attn.out_proj.weight.data
    w5 = model.model.decoder.layers[5].self_attn.out_proj.weight.data
    e3 = model.model.decoder.layers[3].encoder_attn.out_proj.weight.data
    w4[:, 4 * 64:5 * 64].mul_(1.0 - g_gamma)
    w5[:, 1 * 64:2 * 64].mul_(1.0 - g_gamma)
    e3[:, 0 * 64:1 * 64].mul_(1.0 + g_gamma)

    # Edit B: Contrastive sharpening of <|tr|> language token embedding (512 params)
    vocab = tok.get_vocab()
    tr_id = vocab["<|tr|>"]
    other_ids = [
        i for k, i in vocab.items()
        if k.startswith("<|") and k.endswith("|>") and len(k) in (6, 7) and k[2:-2].isalpha() and i != tr_id
    ]
    emb = model.model.decoder.embed_tokens.weight.data
    mu = emb[other_ids].mean(0)
    emb[tr_id].add_(b_beta * (emb[tr_id] - mu))

SAMPLE_RATE = 16000
BLOCK_SEC = 0.10          # 100 ms audio callback blocks
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_SEC)
MIN_SPEECH_SEC = 0.35     # minimum audio length before transcribing
PARTIAL_INTERVAL = 0.55   # seconds between live partial updates while speaking
SILENCE_HOLD_SEC = 0.65   # pause duration to finalize current sentence
MAX_UTTERANCE_SEC = 14.0  # force finalize if speaking continuously for 14s
PRE_ROLL_BLOCKS = 4       # keep 400 ms of audio before speech onset so first consonant isn't clipped

# Common Whisper silent/low-energy hallucinations on Turkish YouTube training data
HALLUCINATION_PHRASES = {
    "altyazı m.k.",
    "altyazı m.k",
    "altyazı mk",
    "izlediğiniz için teşekkürler.",
    "izlediğiniz için teşekkürler",
    "abone olmayı unutmayın.",
    "abone olmayı unutmayın",
    "teşekkürler.",
    "alt yazı",
    "...",
}


class LiveTranscriber:
    def __init__(self, device_idx=None, rms_threshold=0.012, language="turkish"):
        self.device_idx = device_idx
        self.rms_threshold = rms_threshold
        self.language = language

        print(f"[WSP Live] Loading openai/whisper-base on {DEVICE} (fp32)...", flush=True)
        # Load untouched baseline first so we can snapshot & toggle between WSP and Baseline at runtime
        self.proc, self.model = load_model(dtype=torch.float32, state_dict_path=None)
        self.tok = self.proc.tokenizer
        sd_dict = self.model.state_dict()
        self.orig = {k: sd_dict[k].detach().clone() for k in TOUCHED_KEYS}

        # Apply WSP edits by default
        apply_wsp_best(self.model, self.tok)
        self.mode = "WSP"  # "WSP" | "BASELINE" | "COMPARE"
        self.model_lock = threading.Lock()

        # Warm up CUDA kernels so first spoken word has zero lag
        dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self._run_whisper(dummy)
        print("[WSP Live] Model warmed up and ready!", flush=True)

        self.audio_q = queue.Queue()
        self.running = True
        self.current_rms = 0.0
        self.is_speaking = False

        # Callbacks for UI / terminal updates: fn(prev_text, live_text, is_final, latency_ms, mode)
        self.listeners = []
        self.history_line = ""

    def _restore_baseline(self):
        sd_dict = self.model.state_dict()
        for k, v in self.orig.items():
            sd_dict[k].copy_(v)

    def set_mode(self, mode: str):
        """Switch between 'WSP', 'BASELINE', or 'COMPARE'."""
        with self.model_lock:
            self._restore_baseline()
            if mode == "WSP":
                apply_wsp_best(self.model, self.tok)
            self.mode = mode

    def _run_whisper(self, wav_np: np.ndarray) -> str:
        feats = self.proc.feature_extractor(
            wav_np, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(DEVICE, dtype=self.model.dtype)
        with torch.inference_mode():
            ids = self.model.generate(
                feats,
                language=self.language,
                task="transcribe",
                num_beams=1,
                max_new_tokens=128,
            )
        text = self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return text

    def transcribe_buffer(self, wav_np: np.ndarray, is_final: bool = False) -> tuple[str, float]:
        t0 = time.perf_counter()
        with self.model_lock:
            if self.mode == "COMPARE":
                # Run WSP first, then Baseline on the exact same audio buffer
                self._restore_baseline()
                apply_wsp_best(self.model, self.tok)
                wsp_txt = self._run_whisper(wav_np)

                self._restore_baseline()
                base_txt = self._run_whisper(wav_np)
                apply_wsp_best(self.model, self.tok)

                dt_ms = (time.perf_counter() - t0) * 1000.0
                if not wsp_txt and not base_txt:
                    return "", dt_ms
                if wsp_txt.lower() in HALLUCINATION_PHRASES:
                    return "", dt_ms
                return f"WSP : {wsp_txt}\nBASE: {base_txt}", dt_ms
            else:
                txt = self._run_whisper(wav_np)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if txt.lower() in HALLUCINATION_PHRASES:
                    return "", dt_ms
                return txt, dt_ms

    def _audio_callback(self, indata, frames, time_info, status):
        mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        rms = float(np.sqrt(np.mean(mono * mono) + 1e-12))
        self.current_rms = rms
        self.audio_q.put((mono, rms))

    def notify(self, prev_text: str, live_text: str, is_final: bool, latency_ms: float):
        for fn in self.listeners:
            try:
                fn(prev_text, live_text, is_final, latency_ms, self.mode)
            except Exception:
                pass

    def worker_loop(self):
        from collections import deque

        pre_roll = deque(maxlen=PRE_ROLL_BLOCKS)
        utterance_blocks = []
        speaking = False
        silence_sec = 0.0
        last_partial_t = 0.0

        while self.running:
            try:
                block, rms = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if not speaking:
                pre_roll.append(block)
                if rms >= self.rms_threshold:
                    speaking = True
                    self.is_speaking = True
                    silence_sec = 0.0
                    utterance_blocks = list(pre_roll)
                    last_partial_t = time.perf_counter()
            else:
                utterance_blocks.append(block)
                if rms < self.rms_threshold:
                    silence_sec += BLOCK_SEC
                else:
                    silence_sec = 0.0

                dur_sec = len(utterance_blocks) * BLOCK_SEC
                now = time.perf_counter()

                # Check if utterance should be finalized (pause detected or max duration reached)
                if silence_sec >= SILENCE_HOLD_SEC or dur_sec >= MAX_UTTERANCE_SEC:
                    speaking = False
                    self.is_speaking = False
                    if dur_sec >= MIN_SPEECH_SEC:
                        wav = np.concatenate(utterance_blocks, axis=0)
                        # Trim trailing silence blocks slightly for cleaner ending
                        text, dt_ms = self.transcribe_buffer(wav, is_final=True)
                        if text:
                            self.notify(self.history_line, text, True, dt_ms)
                            self.history_line = text
                    utterance_blocks = []
                    pre_roll.clear()

                # Otherwise emit a live streaming partial update while the user is still talking
                elif dur_sec >= MIN_SPEECH_SEC and (now - last_partial_t) >= PARTIAL_INTERVAL:
                    wav = np.concatenate(utterance_blocks, axis=0)
                    text, dt_ms = self.transcribe_buffer(wav, is_final=False)
                    last_partial_t = time.perf_counter()
                    if text:
                        self.notify(self.history_line, text + " …", False, dt_ms)

    def start_stream(self):
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            device=self.device_idx,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self.stream.start()
        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        self.running = False
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


def console_listener(prev_text: str, live_text: str, is_final: bool, latency_ms: float, mode: str):
    tag = f"[{mode} {latency_ms:4.0f}ms]"
    clean_live = live_text.replace("\n", " | ")
    if is_final:
        sys.stdout.write(f"\r\033[K{tag} {clean_live}\n")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\r\033[K{tag} (canlı) {clean_live}")
        sys.stdout.flush()


def run_gui(transcriber: LiveTranscriber):
    import tkinter as tk

    root = tk.Tk()
    root.title("WSP Turkish Live Subtitles")
    root.configure(bg="#111318")
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.92)
    root.overrideredirect(True)  # borderless floating subtitle bar

    # Position near bottom center of screen
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    win_w = min(1000, sw - 80)
    win_h = 165
    pos_x = (sw - win_w) // 2
    pos_y = sh - win_h - 70
    root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")

    # Allow dragging the overlay anywhere with mouse
    drag_state = {"x": 0, "y": 0}

    def start_drag(event):
        drag_state["x"] = event.x
        drag_state["y"] = event.y

    def do_drag(event):
        dx = event.x - drag_state["x"]
        dy = event.y - drag_state["y"]
        x = root.winfo_x() + dx
        y = root.winfo_y() + dy
        root.geometry(f"+{x}+{y}")

    # Top control bar
    top_bar = tk.Frame(root, bg="#1b1f27", height=32)
    top_bar.pack(fill=tk.X, side=tk.TOP)
    top_bar.bind("<Button-1>", start_drag)
    top_bar.bind("<B1-Motion>", do_drag)

    status_dot = tk.Label(
        top_bar, text="● DINLIYOR", fg="#9ca3af", bg="#1b1f27", font=("Segoe UI", 9, "bold")
    )
    status_dot.pack(side=tk.LEFT, padx=(12, 8), pady=4)

    # Mic level canvas
    meter_canvas = tk.Canvas(
        top_bar, width=110, height=10, bg="#2d3340", highlightthickness=0
    )
    meter_canvas.pack(side=tk.LEFT, padx=6, pady=10)
    meter_rect = meter_canvas.create_rectangle(0, 0, 0, 10, fill="#10b981", width=0)

    latency_lbl = tk.Label(
        top_bar, text="Turkish STT (whisper-base)", fg="#6b7280", bg="#1b1f27", font=("Consolas", 9)
    )
    latency_lbl.pack(side=tk.LEFT, padx=10)

    def quit_app(*_):
        transcriber.stop()
        root.destroy()

    quit_btn = tk.Button(
        top_bar,
        text="✕ Kapat (Esc)",
        command=quit_app,
        bg="#374151",
        fg="#f3f4f6",
        activebackground="#ef4444",
        activeforeground="#ffffff",
        relief=tk.FLAT,
        font=("Segoe UI", 8, "bold"),
        padx=8,
        cursor="hand2",
    )
    quit_btn.pack(side=tk.RIGHT, padx=(4, 10), pady=4)

    def clear_subs():
        transcriber.history_line = ""
        prev_lbl.config(text="")
        live_lbl.config(text="Konuşmaya başlayın... (Speak Turkish into your mic)", fg="#9ca3af")

    clear_btn = tk.Button(
        top_bar,
        text="Temizle",
        command=clear_subs,
        bg="#262b36",
        fg="#d1d5db",
        activebackground="#374151",
        relief=tk.FLAT,
        font=("Segoe UI", 8),
        padx=8,
        cursor="hand2",
    )
    clear_btn.pack(side=tk.RIGHT, padx=4, pady=4)

    modes = ["WSP", "BASELINE", "COMPARE"]

    def cycle_mode():
        nxt = modes[(modes.index(transcriber.mode) + 1) % len(modes)]
        transcriber.set_mode(nxt)
        if nxt == "WSP":
            mode_btn.config(text="Mod: WSP (C+G+B)", bg="#065f46", fg="#ecfdf5")
        elif nxt == "BASELINE":
            mode_btn.config(text="Mod: Baseline", bg="#1e3a8a", fg="#eff6ff")
        else:
            mode_btn.config(text="Mod: WSP vs Base", bg="#7c2d12", fg="#fff7ed")

    mode_btn = tk.Button(
        top_bar,
        text="Mod: WSP (C+G+B)",
        command=cycle_mode,
        bg="#065f46",
        fg="#ecfdf5",
        activebackground="#047857",
        activeforeground="#ffffff",
        relief=tk.FLAT,
        font=("Segoe UI", 8, "bold"),
        padx=10,
        cursor="hand2",
    )
    mode_btn.pack(side=tk.RIGHT, padx=4, pady=4)

    # Subtitle body
    body = tk.Frame(root, bg="#111318", padx=20, pady=8)
    body.pack(fill=tk.BOTH, expand=True)
    body.bind("<Button-1>", start_drag)
    body.bind("<B1-Motion>", do_drag)

    prev_lbl = tk.Label(
        body,
        text="",
        fg="#9ca3af",
        bg="#111318",
        font=("Segoe UI", 12),
        wraplength=win_w - 40,
        justify=tk.CENTER,
    )
    prev_lbl.pack(fill=tk.X, pady=(0, 4))
    prev_lbl.bind("<Button-1>", start_drag)
    prev_lbl.bind("<B1-Motion>", do_drag)

    live_lbl = tk.Label(
        body,
        text="Konuşmaya başlayın... (Speak Turkish into your mic)",
        fg="#f9fafb",
        bg="#111318",
        font=("Segoe UI Semibold", 18),
        wraplength=win_w - 40,
        justify=tk.CENTER,
    )
    live_lbl.pack(fill=tk.BOTH, expand=True)
    live_lbl.bind("<Button-1>", start_drag)
    live_lbl.bind("<B1-Motion>", do_drag)

    root.bind("<Escape>", quit_app)

    ui_q = queue.Queue()

    def gui_listener(prev_text: str, live_text: str, is_final: bool, latency_ms: float, mode: str):
        ui_q.put((prev_text, live_text, is_final, latency_ms, mode))

    transcriber.listeners.append(gui_listener)

    def poll_ui():
        # Update mic RMS level bar & speaking indicator
        rms = transcriber.current_rms
        bar_w = min(110, int((rms / max(transcriber.rms_threshold * 5.0, 1e-4)) * 110))
        color = "#10b981" if transcriber.is_speaking else "#4b5563"
        meter_canvas.coords(meter_rect, 0, 0, bar_w, 10)
        meter_canvas.itemconfig(meter_rect, fill=color)

        if transcriber.is_speaking:
            status_dot.config(text="● KONUŞUYOR", fg="#10b981")
        else:
            status_dot.config(text="● DİNLİYOR", fg="#9ca3af")

        while not ui_q.empty():
            prev_text, live_text, is_final, latency_ms, mode = ui_q.get_nowait()
            latency_lbl.config(text=f"{mode} | {latency_ms:.0f} ms")
            if prev_text and prev_text != live_text:
                prev_lbl.config(text=prev_text.replace("\n", "   |   "))
            if is_final:
                live_lbl.config(text=live_text, fg="#ffffff")
            else:
                live_lbl.config(text=live_text, fg="#fde047")  # warm yellow for live streaming partial

        if transcriber.running:
            root.after(40, poll_ui)

    root.after(40, poll_ui)
    root.mainloop()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Live Microphone Turkish Subtitles (WSP Whisper)")
    parser.add_argument("--device", type=int, default=None, help="Microphone input device index (see --list-devices)")
    parser.add_argument("--threshold", type=float, default=0.012, help="Voice activity RMS threshold (default: 0.012)")
    parser.add_argument("--no-gui", action="store_true", help="Run in terminal only without floating subtitle window")
    parser.add_argument("--list-devices", action="store_true", help="List available audio input devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    transcriber = LiveTranscriber(device_idx=args.device, rms_threshold=args.threshold)
    transcriber.listeners.append(console_listener)
    transcriber.start_stream()

    print("\n============================================================")
    print("  [MIC] CANLI TÜRKÇE ALTYAZI AKTİF (WSP Whisper-Base)")
    print("  Mikrofonunuza Türkçe konuşabilirsiniz.")
    print("  Çıkmak için altyazı penceresinde 'Esc' veya Ctrl+C basın.")
    print("============================================================\n", flush=True)

    try:
        if args.no_gui:
            while transcriber.running:
                time.sleep(0.2)
        else:
            run_gui(transcriber)
    except KeyboardInterrupt:
        pass
    finally:
        transcriber.stop()
        print("\n[WSP Live] Durduruldu.")


if __name__ == "__main__":
    main()
