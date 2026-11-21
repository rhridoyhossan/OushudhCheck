from __future__ import annotations

from pathlib import Path
from typing import Optional
import hashlib
import logging
import tempfile

logger = logging.getLogger("oushudhcheck.voice")

DEFAULT_BANGLA_VOICE = "bn-BD-NabanitaNeural"
AUDIO_CACHE_DIR = Path(tempfile.gettempdir()) / "oushudhcheck_tts"
_WHISPER_MODEL = None


def _get_faster_whisper_model():
    """
    Lazy-loads `faster-whisper` (`tiny` / `base` INT8 on CPU) for Bangla & English voice queries
    (per OushudhCheck Hackathon Proposal Section 3.4: `faster-whisper + edge-tts + aiogram v3`).
    """
    global _WHISPER_MODEL
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


async def synthesize_bangla_speech(
    text: str,
    voice: str = DEFAULT_BANGLA_VOICE,
    rate: str = "-5%",
) -> Optional[Path]:
    """
    Synthesizes natural Bangla speech (.mp3) using Microsoft Edge TTS (`edge-tts`).
    Caches generated audio files by text hash for fast replay in web and Telegram bot.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(f"{voice}:{rate}:{cleaned}".encode("utf-8")).hexdigest()[:16]
    out_path = AUDIO_CACHE_DIR / f"tts_{digest}.mp3"

    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info("[DEV-VOICE] Returning cached Bangla TTS audio: %s (%d bytes)", out_path.name, out_path.stat().st_size)
        return out_path

    try:
        import edge_tts  # type: ignore

        logger.info("[DEV-VOICE] Synthesizing Bangla TTS (%s, rate=%s): %r", voice, rate, cleaned[:120])
        communicate = edge_tts.Communicate(text=cleaned, voice=voice, rate=rate)
        await communicate.save(str(out_path))
        logger.info("[DEV-VOICE] Saved Bangla TTS audio -> %s (%d bytes)", out_path, out_path.stat().st_size)
        return out_path
    except Exception as exc:
        logger.error("[DEV-VOICE] Bangla TTS synthesis failed: %s", exc)
        return None
