#!/usr/bin/env bash
# voicespeak — текст -> русский голос, всё внутри Docker.
# Зависимости на машине: только docker + bash.
#
#   ./run.sh build                      # (пере)собрать образ
#   ./run.sh input.txt                  # -> output.wav
#   ./run.sh input.txt result.wav       # свой путь вывода
#   ./run.sh input.txt result.wav --nfe 16   # быстрее (чуть проще качество)
#
set -euo pipefail

IMAGE="voicespeak:latest"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

c()  { printf '\033[1;36m%s\033[0m\n' "$*"; }   # cyan
err(){ printf '\033[1;31m%s\033[0m\n' "$*" >&2; }

command -v docker >/dev/null 2>&1 || { err "нужен docker (не найден)"; exit 1; }

build() {
  c "==> Собираю образ $IMAGE (первый раз долго — тянет torch и вшивает модели)…"
  docker build -t "$IMAGE" "$DIR"
  c "==> Готово."
}

image_exists() { docker image inspect "$IMAGE" >/dev/null 2>&1; }

# детект GPU (общий)
gpu_flags() {
  if command -v nvidia-smi >/dev/null 2>&1 && docker info 2>/dev/null | grep -qiE 'Runtimes:.*nvidia|nvidia'; then
    echo "--gpus all"
  fi
}

# --- режим сборки ---
if [[ "${1:-}" == "build" ]]; then build; exit 0; fi

# --- режим бенчмарка ---
if [[ "${1:-}" == "bench" ]]; then
  shift
  image_exists || build
  read -r -a GPU <<< "$(gpu_flags)"
  [[ ${#GPU[@]} -gt 0 ]] && c "==> GPU обнаружен — бенч на GPU." || c "==> GPU недоступен — бенч на CPU."
  docker run --rm "${GPU[@]}" --entrypoint python "$IMAGE" bench.py "$@"
  exit 0
fi

if [[ $# -lt 1 ]]; then
  err "Использование: ./run.sh <input.txt> [output.wav] [--nfe 16] [--speed 1.0]"
  err "               ./run.sh build   — пересобрать образ"
  exit 1
fi

# --- собрать образ, если его ещё нет ---
image_exists || build

INPUT="$1"; shift
[[ -f "$INPUT" ]] || { err "нет входного файла: $INPUT"; exit 1; }
OUTPUT="output.wav"
EXTRA=()
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then OUTPUT="$1"; shift; fi
EXTRA=("$@")

# абсолютные пути
INPUT_ABS="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")"
OUT_DIR="$(cd "$(dirname "$OUTPUT")" 2>/dev/null && pwd || (mkdir -p "$(dirname "$OUTPUT")" && cd "$(dirname "$OUTPUT")" && pwd))"
OUT_NAME="$(basename "$OUTPUT")"

# --- детект GPU ---
read -r -a GPU <<< "$(gpu_flags)"
if [[ ${#GPU[@]} -gt 0 ]]; then
  c "==> GPU обнаружен — запускаю с ускорением."
else
  c "==> GPU в Docker недоступен — работаю на CPU (медленнее). Для GPU нужен nvidia-container-toolkit."
fi

c "==> Генерация: $(basename "$INPUT")  ->  $OUT_DIR/$OUT_NAME"
docker run --rm "${GPU[@]}" \
  -v "$INPUT_ABS":/data/in.txt:ro \
  -v "$OUT_DIR":/data/out \
  "$IMAGE" /data/in.txt "/data/out/$OUT_NAME" "${EXTRA[@]}"

c "==> Аудио готово: $OUT_DIR/$OUT_NAME"
