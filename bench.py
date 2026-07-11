#!/usr/bin/env python
"""
voicespeak bench — замер скорости инференса на текущей машине.
Гоняет фиксированный текст (детерминированно, seed=42) при nfe=32 и nfe=16,
печатает железо, RTF (время/длительность), xRealtime и пик VRAM.
Запуск:  ./run.sh bench   (или внутри контейнера: python bench.py)
"""
import time, platform, os, argparse

import onnxruntime as ort, numpy as np
_run = ort.InferenceSession.run
def _p(self, o, f, r=None):
    n = {i.name for i in self.get_inputs()}
    if "token_type_ids" in n and "token_type_ids" not in f and "input_ids" in f:
        f = dict(f); f["token_type_ids"] = np.zeros_like(f["input_ids"])
    return _run(self, o, f, r)
ort.InferenceSession.run = _p

import torch
import soundfile as sf
from huggingface_hub import hf_hub_download
from ruaccent import RUAccent
from f5_tts.api import F5TTS

REPO = "Misha24-10/F5-TTS_RUSSIAN"
HERE = os.path.dirname(os.path.abspath(__file__))
REF_WAV = os.path.join(HERE, "assets", "studio_ref.wav")
REF_TXT = open(os.path.join(HERE, "assets", "studio_ref.txt"), encoding="utf-8").read().strip()
RUACCENT_DIR = os.environ.get("RUACCENT_DIR", "/models/ruaccent")

# фиксированный текст (~3 предложения) — одинаковая длительность на всех машинах
BENCH_TEXT = ("Утро выдалось тихим и прохладным, за окном медленно таял туман. "
              "Она налила себе чашку кофе и села у окна, глядя на пустую улицу. "
              "Впереди её ждал долгий и непростой день.")


def hr(): print("-" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2, help="таймингов на конфигурацию (берётся лучший)")
    ap.add_argument("--nfe", type=int, nargs="*", default=[32, 16], help="какие nfe мерить")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    hr(); print(" voicespeak — БЕНЧМАРК ИНФЕРЕНСА"); hr()
    print(f" ОС:        {platform.system()} {platform.release()} / {platform.machine()}")
    print(f" CPU ядер:  {os.cpu_count()}")
    print(f" torch:     {torch.__version__}")
    print(f" устройство:{device.upper()}")
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f" GPU:       {p.name} | {p.total_memory/1e9:.1f} GB VRAM")
    hr()

    acc = RUAccent(); acc.load(omograph_model_size="turbo", use_dictionary=True, workdir=RUACCENT_DIR)
    ref_text = acc.process_all(REF_TXT)
    gen_text = acc.process_all(BENCH_TEXT)

    ckpt = hf_hub_download(REPO, "F5TTS_v1_Base_accent_tune/model_last_inference.safetensors")
    vocab = hf_hub_download(REPO, "F5TTS_v1_Base/vocab.txt")
    f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt, vocab_file=vocab, device=device)
    f5.ema_model = f5.ema_model.float()
    try:
        f5.vocoder = f5.vocoder.float()
    except Exception:
        pass

    def synth(nfe):
        out = "/tmp/_bench.wav"
        f5.infer(ref_file=REF_WAV, ref_text=ref_text, gen_text=gen_text,
                 file_wave=out, nfe_step=nfe, seed=42)
        if device == "cuda":
            torch.cuda.synchronize()
        a = sf.read(out)[0]; return len(a) / sf.info(out).samplerate

    # прогрев
    print(" прогрев…", flush=True)
    synth(8)

    results = []
    for nfe in args.nfe:
        best = None; dur = None
        for _ in range(args.runs):
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time(); dur = synth(nfe); dt = time.time() - t0
            best = dt if best is None else min(best, dt)
        vram = torch.cuda.max_memory_allocated()/1e9 if device == "cuda" else None
        rtf = best / dur
        results.append((nfe, dur, best, rtf, vram))
        vtxt = f" | VRAM {vram:.2f} GB" if vram else ""
        print(f" nfe={nfe:>2}: {best:6.1f}s счёта на {dur:4.1f}s аудио  ->  RTF {rtf:4.2f}  ({1/rtf:5.2f}x реалтайм){vtxt}", flush=True)

    hr()
    # строка для обмена результатами между машинами
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else f"CPU x{os.cpu_count()}"
    tag = ";".join(f"nfe{n}=RTF{r:.2f}" for n, _, _, r, _ in results)
    print(f" ИТОГ: [{gpu}] {tag}")
    print(" (RTF меньше = быстрее; <1.0 = быстрее реального времени)")
    hr()


if __name__ == "__main__":
    main()
