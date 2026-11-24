from __future__ import annotations

from pathlib import Path
from typing import Optional
import asyncio
import hashlib
import logging
import os
import re
import tempfile
import threading
import uuid

logger = logging.getLogger("oushudhcheck.voice")

DEFAULT_BANGLA_VOICE = "bn-BD-NabanitaNeural"
AUDIO_CACHE_DIR = Path(tempfile.gettempdir()) / "oushudhcheck_tts"
MAX_TTS_CACHE_FILES = 200
MAX_TTS_CACHE_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_TTS_TEXT_CHARS = 6000
_AUDIO_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_WHISPER_MODEL = None
_WHISPER_LOCK = threading.Lock()


def _evict_tts_cache_if_needed() -> None:
    """Keep the TTS disk cache bounded by file count and total size."""
    try:
        files = [
            p for p in AUDIO_CACHE_DIR.glob("tts_*.mp3")
            if p.is_file() and ".tmp." not in p.name
        ]
        if len(files) <= MAX_TTS_CACHE_FILES:
            total_bytes = sum(p.stat().st_size for p in files)
            if total_bytes <= MAX_TTS_CACHE_BYTES:
                return
        files.sort(key=lambda p: p.stat().st_mtime)
        total_bytes = sum(p.stat().st_size for p in files)
        while files and (len(files) > MAX_TTS_CACHE_FILES or total_bytes > MAX_TTS_CACHE_BYTES):
            oldest = files.pop(0)
            try:
                sz = oldest.stat().st_size
                oldest.unlink(missing_ok=True)
                total_bytes -= sz
            except OSError:
                pass
    except OSError:
        pass


def _get_faster_whisper_model():
    """
    Lazy-loads `faster-whisper` (`tiny` / `base` INT8 on CPU) for Bangla & English voice queries.
    """
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    with _WHISPER_LOCK:
        if _WHISPER_MODEL is not None:
            return _WHISPER_MODEL
        try:
            from faster_whisper import WhisperModel  # type: ignore

            logger.info("[DEV-VOICE] Loading faster-whisper ('base', compute_type='int8') speech recognition model...")
            _WHISPER_MODEL = WhisperModel("base", device="cpu", compute_type="int8")
            logger.info("[DEV-VOICE] Loaded faster-whisper ASR model successfully.")
            return _WHISPER_MODEL
        except Exception as exc:
            logger.warning("[DEV-VOICE] faster-whisper unavailable: %s", exc)
            return None


_NON_BANGLA_INDIC_RE = re.compile(r"[\u0900-\u097F\u0600-\u06FF\u0A00-\u0A7F]")
_MEANINGFUL_TEXT_RE = re.compile(r"[\u0980-\u09FFA-Za-z0-9]")


def _collect_valid_segments(segments) -> str:
    """
    Collects transcribed segments while filtering out high-no-speech noise artifacts,
    corrupted UTF-8 replacement sequences (`\\ufffd`), and immediate repetitions.
    """
    parts: list[str] = []
    prev_norm = ""
    for seg in segments:
        seg_text = (getattr(seg, "text", "") or "").strip()
        if not seg_text or "\ufffd" in seg_text or not _MEANINGFUL_TEXT_RE.search(seg_text):
            continue
        no_speech = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
        avg_logprob = float(getattr(seg, "avg_logprob", 0.0) or 0.0)
        if no_speech >= 0.85 and avg_logprob < -1.0:
            continue
        norm = re.sub(r"\s+", " ", seg_text).strip().lower()
        if norm and norm == prev_norm:
            continue
        parts.append(seg_text)
        prev_norm = norm

    text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    if not text or "\ufffd" in text or not _MEANINGFUL_TEXT_RE.search(text):
        return ""
    return text


def _decode_audio_to_16k_mono(audio_path: Path):
    """
    Decodes an audio file (.ogg / .mp3 / .wav / .m4a) into a 16 kHz mono float32 1D numpy array
    using PyAV directly (avoiding `faster_whisper.audio.decode_audio()` passing `metadata_errors`,
    which PyAV >= 19 removed). Returns `None` if decoding yields no samples, or falls back to
    `str(audio_path)` only when `av` is unavailable.
    """
    try:
        import av  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return str(audio_path)

    resampler = av.audio.resampler.AudioResampler(
        format="flt",
        layout="mono",
        rate=16000,
    )
    chunks = []
    with av.open(str(audio_path), mode="r") as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                arr = resampled.to_ndarray()
                chunks.append(np.asarray(arr, dtype=np.float32).reshape(-1))
        # Flush resampler tail
        for resampled in resampler.resample(None):
            arr = resampled.to_ndarray()
            chunks.append(np.asarray(arr, dtype=np.float32).reshape(-1))

    if not chunks:
        return None
    audio_np = np.concatenate(chunks, axis=0)
    if audio_np.size == 0:
        return None
    return np.ascontiguousarray(audio_np, dtype=np.float32)


def _transcribe_with_google_bn_bd(audio_np) -> Optional[str]:
    """
    Primary Bangla & Bangla-English ASR using `SpeechRecognition` (`bn-BD`) on a 16 kHz mono
    float32 numpy array. Returns `None` if unavailable/offline/unrecognized so caller can fall
    back to local `faster-whisper`.
    """
    if isinstance(audio_np, str):
        return None
    try:
        import numpy as np  # type: ignore
        import speech_recognition as sr  # type: ignore
    except ImportError:
        return None

    try:
        pcm16 = (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype("<i2")
        audio_data = sr.AudioData(pcm16.tobytes(), 16000, 2)
        recognizer = sr.Recognizer()
        text = (recognizer.recognize_google(audio_data, language="bn-BD") or "").strip()
        if text and "\ufffd" not in text and _MEANINGFUL_TEXT_RE.search(text):
            logger.info("[DEV-VOICE] Google bn-BD ASR transcribed: %r", text)
            return text
        return None
    except Exception as exc:
        logger.debug("[DEV-VOICE] Google bn-BD ASR unavailable/unrecognized (%s); falling back to faster-whisper.", exc)
        return None


def _run_whisper_pass(
    model,
    audio_input,
    *,
    language: Optional[str],
    use_vad: bool,
):
    kwargs = {
        "beam_size": 3,
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.65,
        "vad_filter": use_vad,
    }
    if language:
        kwargs["language"] = language
    if use_vad:
        kwargs["vad_parameters"] = {
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 300,
        }
    segments, info = model.transcribe(audio_input, **kwargs)
    return _collect_valid_segments(segments), info


def transcribe_bangla_audio(audio_path: Path) -> Optional[str]:
    """
    Transcribes a Bangla or Bangla-English voice question (.ogg / .wav / .mp3) into text.
    Uses PyAV 16 kHz mono decoding + `bn-BD` speech recognition for accurate native Bangla
    transcription (~1s), with automatic offline fallback to `faster-whisper` (`base` INT8 CPU).
    """
    if not audio_path:
        return None
    path_obj = Path(audio_path) if not isinstance(audio_path, Path) else audio_path
    try:
        if not path_obj.exists() or not path_obj.is_file() or path_obj.stat().st_size == 0:
            logger.warning("[DEV-VOICE] Audio file missing or empty: %s", path_obj)
            return None
    except OSError as exc:
        logger.warning("[DEV-VOICE] Cannot stat audio file %s: %s", path_obj, exc)
        return None

    try:
        audio_input = _decode_audio_to_16k_mono(path_obj)
        if audio_input is None:
            logger.warning("[DEV-VOICE] Audio file contained no decodable samples: %s", path_obj)
            return None

        # 1. Primary: accurate bn-BD recognizer (fast, handles Bangla + English medicine names cleanly)
        bn_online = _transcribe_with_google_bn_bd(audio_input)
        if bn_online:
            return bn_online

        # 2. Offline fallback: local faster-whisper (without Bangla initial_prompt which causes UTF-8 corruption on base)
        model = _get_faster_whisper_model()
        if model is None:
            return None

        text = ""
        info = None
        try:
            text, info = _run_whisper_pass(model, audio_input, language=None, use_vad=True)
        except Exception as vad_exc:
            logger.debug("[DEV-VOICE] VAD pass failed (%s); retrying without VAD.", vad_exc)

        if not text:
            text, info = _run_whisper_pass(model, audio_input, language=None, use_vad=False)

        detected_lang = (getattr(info, "language", "") or "").lower()
        lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)

        if text and _NON_BANGLA_INDIC_RE.search(text):
            logger.warning("[DEV-VOICE] Rejecting non-Bangla Indic/Arabic script fallback output (%s): %r", detected_lang, text)
            return None

        logger.info(
            "[DEV-VOICE] faster-whisper fallback transcribed (%s, prob=%.2f): %r",
            detected_lang or "unknown",
            lang_prob,
            text,
        )
        return text or None
    except Exception as exc:
        logger.error("[DEV-VOICE] Voice transcription error: %s", exc)
        return None


_NUMERIC_RANGE_RE = re.compile(
    r"([০-৯0-9]+(?:\.[০-৯0-9]+)?)\s*[-–—]\s*([০-৯0-9]+(?:\.[০-৯0-9]+)?)"
)
_TTS_STRIP_RE = re.compile(r"[💊⚠️🚨✅📖🔍🔊🎤🗣️⏳•*#_`~<>]+")


def normalize_bangla_tts_text(text: str, max_chars: int = MAX_TTS_TEXT_CHARS) -> str:
    """
    Normalizes Bangla/English clinical text before Edge TTS synthesis so numeric ranges
    like '৩-৪' or '4-6' are spoken naturally as '৩ থেকে ৪' / '4 থেকে 6' instead of 'minus',
    and punctuation/symbols produce smooth pauses without mid-sentence truncation.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = _NUMERIC_RANGE_RE.sub(r"\1 থেকে \2", cleaned)
    cleaned = _TTS_STRIP_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("—", ", ").replace("–", " থেকে ").replace("|", ", ")
    cleaned = cleaned.replace("()", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > max_chars:
        truncated = cleaned[:max_chars]
        last_stop = max(truncated.rfind("।"), truncated.rfind("."))
        cleaned = (truncated[: last_stop + 1] if last_stop > max_chars // 2 else truncated).strip()
    return cleaned


def get_tts_cache_key(
    text: str,
    voice: str = DEFAULT_BANGLA_VOICE,
    rate: str = "-5%",
) -> str:
    cleaned = normalize_bangla_tts_text(text)
    return hashlib.sha256(f"{voice}:{rate}:{cleaned}".encode("utf-8")).hexdigest()[:16]


def get_cached_tts_audio_path(audio_id: str) -> Optional[Path]:
    """Returns the cached MP3 path for a valid 16-hex-char audio ID if it exists on disk."""
    if not audio_id or not _AUDIO_ID_RE.match(audio_id):
        return None
    candidate = AUDIO_CACHE_DIR / f"tts_{audio_id}.mp3"
    if candidate.exists() and candidate.stat().st_size > 0:
        return candidate
    return None


async def synthesize_bangla_speech(
    text: str,
    voice: str = DEFAULT_BANGLA_VOICE,
    rate: str = "-5%",
) -> Optional[Path]:
    """
    Synthesizes natural Bangla speech (.mp3) using Microsoft Edge TTS (`edge-tts`).
    Caches generated audio files by text hash for fast replay in web and Telegram bot.
    """
    cleaned = normalize_bangla_tts_text(text)
    if not cleaned:
        return None

    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = get_tts_cache_key(cleaned, voice=voice, rate=rate)
    out_path = AUDIO_CACHE_DIR / f"tts_{digest}.mp3"

    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info("[DEV-VOICE] Returning cached Bangla TTS audio: %s (%d bytes)", out_path.name, out_path.stat().st_size)
        return out_path

    tmp_path = AUDIO_CACHE_DIR / f"tts_{digest}.{uuid.uuid4().hex[:8]}.tmp.mp3"
    try:
        import edge_tts  # type: ignore

        logger.info("[DEV-VOICE] Synthesizing Bangla TTS (%s, rate=%s): %r", voice, rate, cleaned[:120])
        communicate = edge_tts.Communicate(text=cleaned, voice=voice, rate=rate)
        await communicate.save(str(tmp_path))
        os.replace(tmp_path, out_path)
        await asyncio.to_thread(_evict_tts_cache_if_needed)
        logger.info("[DEV-VOICE] Saved Bangla TTS audio -> %s (%d bytes)", out_path, out_path.stat().st_size)
        return out_path
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        logger.error("[DEV-VOICE] Bangla TTS synthesis failed: %s", exc)
        return None
