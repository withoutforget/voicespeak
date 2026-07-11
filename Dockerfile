# voicespeak — самодостаточный образ русского TTS (F5-TTS).
# CUDA-сборка torch: использует GPU при запуске с --gpus all, иначе работает на CPU.
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/models \
    RUACCENT_DIR=/models/ruaccent \
    PYTHONUNBUFFERED=1

# системные зависимости: ffmpeg (нормализация), libsndfile (soundfile), git (часть pip-зависимостей)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# python-зависимости (torch отдельно — CUDA-колесо большое)
RUN pip install --no-cache-dir torch torchaudio \
    && pip install --no-cache-dir f5-tts ruaccent onnxruntime soundfile huggingface_hub

WORKDIR /app
COPY speak.py /app/speak.py
COPY bench.py /app/bench.py
COPY assets /app/assets

# ВШИВАЕМ модели в образ (F5 + вокодер vocos + RUAccent) — чтобы запуск был офлайн и быстрый
RUN python - <<'PY'
import onnxruntime as ort, numpy as np
_run = ort.InferenceSession.run
def _p(self, o, f, r=None):
    n = {i.name for i in self.get_inputs()}
    if "token_type_ids" in n and "token_type_ids" not in f and "input_ids" in f:
        f = dict(f); f["token_type_ids"] = np.zeros_like(f["input_ids"])
    return _run(self, o, f, r)
ort.InferenceSession.run = _p
from huggingface_hub import hf_hub_download
from ruaccent import RUAccent
from f5_tts.api import F5TTS
REPO = "Misha24-10/F5-TTS_RUSSIAN"
ckpt = hf_hub_download(REPO, "F5TTS_v1_Base_accent_tune/model_last_inference.safetensors")
vocab = hf_hub_download(REPO, "F5TTS_v1_Base/vocab.txt")
# инициализация на CPU скачивает vocos-вокодер в кэш образа
f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt, vocab_file=vocab, device="cpu")
# модели RUAccent
acc = RUAccent(); acc.load(omograph_model_size="turbo", use_dictionary=True, workdir="/models/ruaccent")
print("models baked OK")
PY

# теперь всё в кэше — запуск полностью офлайн
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

ENTRYPOINT ["python", "speak.py"]
