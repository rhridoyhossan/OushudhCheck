from __future__ import annotations

from pathlib import Path
from typing import Optional
import hashlib
import logging
import tempfile

logger = logging.getLogger("oushudhcheck.voice")

DEFAULT_BANGLA_VOICE = "bn-BD-NabanitaNeural"
AUDIO_CACHE_DIR = Path(tempfile.gettempdir()) / "oushudhcheck_tts"


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
