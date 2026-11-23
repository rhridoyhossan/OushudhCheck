#!/bin/sh
set -eu

MODEL_DIR="${MODEL_DIR:-/models}"
MODEL_FILENAME="${MODEL_FILENAME:-oushudhcheck-gemma3-1b-bn.Q4_K_M.gguf}"
MODEL_RELEASE_URL="${MODEL_RELEASE_URL:-https://github.com/rhridoyhossan/OushudhCheck/releases/latest/download/oushudhcheck_gemma3_1b_deploy.zip}"

TARGET_GGUF="${MODEL_DIR}/${MODEL_FILENAME}"
mkdir -p "${MODEL_DIR}"

if [ -f "${TARGET_GGUF}" ] && [ "$(wc -c < "${TARGET_GGUF}")" -gt 104857600 ]; then
  echo "[model-init] Model already exists at ${TARGET_GGUF} ($(du -h "${TARGET_GGUF}" | cut -f1)). Skipping download."
  exit 0
fi

echo "[model-init] Downloading Gemma 3 1B model from:"
echo "  ${MODEL_RELEASE_URL}"

TMP_FILE="${MODEL_DIR}/download.tmp"
rm -f "${TMP_FILE}"

curl -fL --retry 3 --retry-delay 5 --progress-bar -o "${TMP_FILE}" "${MODEL_RELEASE_URL}"

case "${MODEL_RELEASE_URL}" in
  *.gguf|*.gguf\?*)
    mv "${TMP_FILE}" "${TARGET_GGUF}"
    ;;
  *)
    echo "[model-init] Extracting model artifacts into ${MODEL_DIR}..."
    TMP_EXTRACT="${MODEL_DIR}/.extract_tmp"
    rm -rf "${TMP_EXTRACT}"
    mkdir -p "${TMP_EXTRACT}"
    unzip -o -j "${TMP_FILE}" "*.gguf" "system_prompt.txt" "Modelfile" -d "${TMP_EXTRACT}" || \
      unzip -o -j "${TMP_FILE}" -d "${TMP_EXTRACT}"
    rm -f "${TMP_FILE}"

    FOUND_GGUF="$(find "${TMP_EXTRACT}" -maxdepth 2 -type f -name '*.gguf' | head -n 1)"
    if [ -z "${FOUND_GGUF}" ]; then
      echo "[model-init] ERROR: No .gguf file found inside archive!" >&2
      rm -rf "${TMP_EXTRACT}"
      exit 1
    fi

    mv "${FOUND_GGUF}" "${TARGET_GGUF}"
    if [ -f "${TMP_EXTRACT}/system_prompt.txt" ]; then
      mv "${TMP_EXTRACT}/system_prompt.txt" "${MODEL_DIR}/system_prompt.txt"
    fi
    if [ -f "${TMP_EXTRACT}/Modelfile" ]; then
      mv "${TMP_EXTRACT}/Modelfile" "${MODEL_DIR}/Modelfile"
    fi
    rm -rf "${TMP_EXTRACT}"
    ;;
esac

echo "[model-init] Ready: ${TARGET_GGUF} ($(du -h "${TARGET_GGUF}" | cut -f1))"
