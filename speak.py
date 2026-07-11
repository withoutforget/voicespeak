#!/usr/bin/env python
"""
voicespeak — русский TTS-голос, локально.
Вход: текстовый файл. Выход: wav. Рецепт: F5-TTS accent-tune + RUAccent + студийный референс + громкая нормализация.
Модели вшиты в образ (офлайн). Устройство определяется автоматически (CUDA -> иначе CPU).
"""
import sys, os, time, argparse, subprocess

# --- патч onnxruntime для RUAccent (token_type_ids) ---
import onnxruntime as ort, numpy as np
_run = ort.InferenceSession.run
def _patched(self, outs, feed, ro=None):
    need = {i.name for i in self.get_inputs()}
    if "token_type_ids" in need and "token_type_ids" not in feed and "input_ids" in feed:
        feed = dict(feed); feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
    return _run(self, outs, feed, ro)
ort.InferenceSession.run = _patched

import torch
from huggingface_hub import hf_hub_download
from ruaccent import RUAccent
from f5_tts.api import F5TTS

REPO = "Misha24-10/F5-TTS_RUSSIAN"
CKPT_REL = "F5TTS_v1_Base_accent_tune/model_last_inference.safetensors"
VOCAB_REL = "F5TTS_v1_Base/vocab.txt"
HERE = os.path.dirname(os.path.abspath(__file__))
REF_WAV = os.path.join(HERE, "assets", "studio_ref.wav")
REF_TXT_FILE = os.path.join(HERE, "assets", "studio_ref.txt")
RUACCENT_DIR = os.environ.get("RUACCENT_DIR", "/models/ruaccent")


def log(msg):
    print(f"[voicespeak] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="текстовый файл на входе")
    ap.add_argument("output", nargs="?", default="/data/out.wav", help="wav на выходе")
    ap.add_argument("--nfe", type=int, default=32, help="шагов диффузии: 32=качество, 16=быстрее")
    ap.add_argument("--speed", type=float, default=1.0, help="скорость речи")
    args = ap.parse_args()

    text = open(args.input, encoding="utf-8").read().strip()
    if not text:
        log("пустой входной файл"); sys.exit(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"устройство: {device.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " (GPU не найден — будет медленно)"))
    log(f"символов на входе: {len(text)} | nfe={args.nfe}")

    t0 = time.time()
    log("ставлю ударения (RUAccent)…")
    acc = RUAccent()
    acc.load(omograph_model_size="turbo", use_dictionary=True, workdir=RUACCENT_DIR)
    ref_text = acc.process_all(open(REF_TXT_FILE, encoding="utf-8").read().strip())
    gen_text = acc.process_all(text)

    log("гружу модель F5…")
    ckpt = hf_hub_download(REPO, CKPT_REL)
    vocab = hf_hub_download(REPO, VOCAB_REL)
    f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt, vocab_file=vocab, device=device)
    f5.ema_model = f5.ema_model.float()  # FP32: корректно на всех картах (в т.ч. GTX 1660, где FP16=NaN)
    try:
        f5.vocoder = f5.vocoder.float()
    except Exception:
        pass

    log("синтез…")
    raw = args.output + ".raw.wav"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    f5.infer(ref_file=REF_WAV, ref_text=ref_text, gen_text=gen_text,
             file_wave=raw, nfe_step=args.nfe, speed=args.speed, seed=42)

    log("нормализую громкость…")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw,
                    "-af", "loudnorm=I=-14:TP=-1.0", args.output], check=True)
    os.remove(raw)

    try:
        import soundfile as sf
        dur = len(sf.read(args.output)[0]) / sf.info(args.output).samplerate
        dt = time.time() - t0
        log(f"готово: {args.output} | {dur:.1f}с аудио за {dt:.1f}с (RTF {dt/dur:.2f})")
    except Exception:
        log(f"готово: {args.output}")


if __name__ == "__main__":
    main()
