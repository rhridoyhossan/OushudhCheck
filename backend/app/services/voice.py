from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple
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
_TTS_STRIP_SYMBOLS_RE = re.compile(r"[⚠️🚨💊✅🔊📖•]+")
_WHISPER_MODEL = None
_WHISPER_LOCK = threading.Lock()


def prepare_bangla_tts_text(text: str, max_chars: int = MAX_TTS_TEXT_CHARS) -> str:
    """
    Normalizes Bangla/mixed medicine summaries for natural Edge Neural TTS pronunciation.
    Strips UI symbols/emojis and converts line breaks into clean spoken pauses while
    preserving the complete medicine summary, dosage instructions, and warnings.
    """
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return ""

    raw = _TTS_STRIP_SYMBOLS_RE.sub(" ", raw)
    raw = re.sub(r"\s*[—–]\s*", ", ", raw)

    normalized_lines = []
    for line in raw.split("\n"):
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        if not cleaned_line:
            continue
        if cleaned_line[-1] not in "।.!?,;:":
            cleaned_line += "।"
        normalized_lines.append(cleaned_line)

    cleaned = re.sub(r"\s+", " ", " ".join(normalized_lines)).strip()
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
    cleaned = prepare_bangla_tts_text(text)
    return hashlib.sha256(f"{voice}:{rate}:{cleaned}".encode("utf-8")).hexdigest()[:16]


def get_cached_tts_audio_path(audio_id: str) -> Optional[Path]:
    """Returns the cached MP3 path for a valid 16-hex-char audio ID if it exists on disk."""
    if not audio_id or not _AUDIO_ID_RE.match(audio_id):
        return None
    candidate = AUDIO_CACHE_DIR / f"tts_{audio_id}.mp3"
    if candidate.exists() and candidate.stat().st_size > 0:
        return candidate
    return None


def build_scan_bangla_voice_script(
    matched_strips: Sequence[Any],
    guidance_list: Sequence[Tuple[str, str, str]],
    safety_alerts: Sequence[Any],
    fallback_summary_bn: str = "",
) -> str:
    """
    Builds the complete Bangla voice script for scanned medicine strips, including each
    medicine's full `সহজ বাংলা সারসংক্ষেপ` (indications, usage/timing, and warnings)
    followed by duplicate/interaction safety alerts.
    """
    if not matched_strips:
        return (fallback_summary_bn or "").strip()

    voice_parts = [f"আপনার ছবিতে {len(matched_strips)}টি ওষুধের পাতা শনাক্ত করা হয়েছে।"]
    for idx, (strip, guidance) in enumerate(zip(matched_strips, guidance_list), start=1):
        med = getattr(strip, "matched_medicine", None)
        if not med:
            continue
        dosage_bn, bangla_summary, _timing_tag = guidance
        summary_clean = " ".join((bangla_summary or "").split()).strip()
        if summary_clean:
            voice_parts.append(f"স্ট্রিপ {idx}: {summary_clean}")
        else:
            strength = getattr(strip, "extracted_strength", None) or ""
            strength_part = f" {strength}" if strength else ""
            fallback_line = f"স্ট্রিপ {idx}: {med.brand_name}{strength_part}, জেনেরিক {med.generic}।"
            if dosage_bn:
                fallback_line = f"{fallback_line} {dosage_bn}"
            voice_parts.append(fallback_line)

    for alert in safety_alerts:
        msg_bn = (getattr(alert, "message_bn", "") or "").strip()
        if msg_bn:
            voice_parts.append(msg_bn)

    if not safety_alerts:
        voice_parts.append("ওষুধগুলোতে কোনো ডুপ্লিকেট বা ঝুঁকিপূর্ণ আন্তঃক্রিয়া পাওয়া যায়নি।")

    return "\n".join(voice_parts)


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


def transcribe_bangla_audio(audio_path: Path) -> Optional[str]:
    """
    Transcribes a Bangla or English voice question (.ogg / .wav / .mp3) into text using `faster-whisper`.
    """
    if not audio_path or not audio_path.exists():
        return None

    model = _get_faster_whisper_model()
    if model is None:
        return None

    try:
        segments, info = model.transcribe(str(audio_path), beam_size=3)
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
        logger.info(
            "[DEV-VOICE] faster-whisper transcribed (%s, prob=%.2f): %r",
            getattr(info, "language", "bn"),
            getattr(info, "language_probability", 1.0),
            text,
        )
        return text or None
    except Exception as exc:
        logger.error("[DEV-VOICE] faster-whisper transcription error: %s", exc)
        return None


_NUMERIC_RANGE_RE = re.compile(
    r"([০-৯0-9]+(?:\.[০-৯0-9]+)?)\s*[-–—]\s*([০-৯0-9]+(?:\.[০-৯0-9]+)?)"
)
_TTS_STRIP_RE = re.compile(r"[💊⚠️🚨✅📖🔍🔊🎤🗣️⏳•*#_`~<>]+")


def normalize_bangla_tts_text(text: str) -> str:
    """
    Normalizes Bangla/English clinical text before Edge TTS synthesis so numeric ranges
    like '৩-৪' or '4-6' are spoken naturally as '৩ থেকে ৪' / '4 থেকে 6' instead of 'minus',
    and punctuation/symbols produce smooth pauses.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = _NUMERIC_RANGE_RE.sub(r"\1 থেকে \2", cleaned)
    cleaned = _TTS_STRIP_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("—", ", ").replace("–", " থেকে ").replace("|", ", ")
    cleaned = cleaned.replace("()", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


async def synthesize_bangla_speech(
    text: str,
    voice: str = DEFAULT_BANGLA_VOICE,
    rate: str = "-5%",
) -> Optional[Path]:
    """
    Synthesizes natural Bangla speech (.mp3) using Microsoft Edge TTS (`edge-tts`).
    Caches generated audio files by text hash for fast replay in web and Telegram bot.
    """
<<<<<<< HEAD
    cleaned = prepare_bangla_tts_text(text)
=======
    cleaned = normalize_bangla_tts_text(text)
>>>>>>> bdab9a43f44c711a1dc7d0452e9326bb4a0da795
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
