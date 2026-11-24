"""
OushudhCheck Telegram Bot (@OushudhCheck_bot)
Multi-Strip Medicine Safety Scanner & Bangla Voice Medication Assistant.

Features:
- Loads TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_USERNAME from .env
- Live 5-Stage Progress Feedback (`editMessageText` + `sendChatAction`) so the user sees every pipeline step in real time
- Bangla Summary (`সহজ বাংলা সারসংক্ষেপ`), schedule, and duplicate warnings
- Supports photo scans (1–6 strips), Caregiver Mode follow-up Q&A (voice & text with 30-min session memory), Bangla voice queries (`faster-whisper` ASR), and text queries
- Replies with Bangla spoken voice audio (`edge-tts` `bn-BD-NabanitaNeural`)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import asyncio
import html
import logging
import os
import re
import sys
import tempfile
import threading
import time
import uuid

import aiohttp
import cv2
import numpy as np
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

from backend.app.models.schemas import SafetyAlert, StripScanResult
from backend.app.services.ocr import MedicineStripScanner, evaluate_multi_strip_safety
from backend.app.services.slm import answer_caregiver_followup
from backend.app.services.voice import synthesize_bangla_speech, transcribe_bangla_audio

MAX_TELEGRAM_MSG_LEN = 3900
MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
CAREGIVER_SESSION_TTL_SEC = 30 * 60  # 30 minutes

_FOLLOWUP_SHORT_WORD_RE = re.compile(
    r"(?:^|[\s,।?!;:\"'()\-])(কি|কী|কেন|কত)(?=$|[\s,।?!;:\"'()\-])"
)
_FOLLOWUP_KEYWORD_RE = re.compile(
    r"(?:"
    r"কখন|কতবার|কীভাবে|কিভাবে|খাওয়া|খাবে|খাবো|খেলে|খেতে|পারবো|পারব|খাবারের|খাবার|"
    r"সকালে|দুপুরে|রাতে|মেডিসিন|জেনেরিক|উপাদান|"
    r"আগে|পরে|খালি\s*পেটে|ভরা\s*পেটে|পার্শ্বপ্রতিক্রিয়া|পার্শ্ব\s*প্রতিক্রিয়া|"
    r"ক্ষতি|সমস্যা|নিরাপদ|গর্ভাবস্থ|গর্ভবতী|স্তন্যদান|বুকের\s*দুধ|সতর্কতা|নিষেধ|"
    r"সংরক্ষণ|রাখব|রাখা|কাজ|অন্য\s*ওষুধ|একসাথে|ডোজ|মাত্রা|এই\s*ওষুধ|"
    r"ওষুধটি|ওষুধটা|এটা|এটি|দিতে\s*হবে|খাওয়াবো|বাচ্চা|শিশু|বয়স্ক|"
    r"\b(?:before\s+meal|after\s+meal|empty\s+stomach|with\s+food|food|eat|"
    r"side\s+effects?|pregnant|pregnancy|breastfeed\w*|lactation|safe|safety|"
    r"interact\w*|together|warning|precaution|storage|store|how\s+to|when\s+to|"
    r"how\s+many|how\s+often|dosage|dose|what\s+is|used\s+for|can\s+i|should\s+i|give)\b"
    r")",
    flags=re.IGNORECASE,
)


def _looks_like_followup_question(text: str) -> bool:
    """Detect whether a voice/text utterance is a caregiver follow-up question rather than a standalone brand name."""
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if "?" in cleaned or "؟" in cleaned:
        return True
    if _FOLLOWUP_KEYWORD_RE.search(cleaned) or _FOLLOWUP_SHORT_WORD_RE.search(cleaned):
        return True
    return False


def _esc(val: Any) -> str:
    return html.escape(str(val), quote=False)


def _format_tg_summary(summary: str, indent: str = "") -> str:
    """Escapes summary for Telegram HTML and always places 'সতর্কতা:' on a clean new line."""
    escaped = _esc(summary.strip())
    return re.sub(
        r"(?<!নিরাপত্তা)[ \t]*\n?[ \t]*(?<!নিরাপত্তা )(সতর্কতা\s*:)",
        f"\n{indent}⚠️ <b>সতর্কতা:</b>",
        escaped,
    )


def _chunk_html_blocks(blocks: List[str], max_len: int = MAX_TELEGRAM_MSG_LEN) -> List[str]:
    chunks: List[str] = []
    current = ""
    for block in blocks:
        candidate = f"{current}\n{block}" if current else block
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(block) <= max_len:
                current = block
            else:
                plain = html.unescape(re.sub(r"<[^>]+>", "", block))
                budget = max(0, max_len - 3)
                cut = min(len(plain), budget)
                safe_truncated = _esc(plain[:cut])
                while cut > 0 and len(safe_truncated) > budget:
                    cut = max(0, cut - max(1, len(safe_truncated) - budget))
                    safe_truncated = _esc(plain[:cut])
                chunks.append(safe_truncated + "...")
                current = ""
    if current:
        chunks.append(current)
    return chunks or [""]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("oushudhcheck.bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_BOT_USERNAME = os.getenv("TELEGRAM_BOT_USERNAME", "OushudhCheck_bot").strip()
API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
FILE_BASE = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}"


class OushudhCheckTelegramBot:
    def __init__(self) -> None:
        if not TELEGRAM_BOT_TOKEN:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is missing in .env")
        self.scanner = MedicineStripScanner()
        self.offset = 0
        self.caregiver_sessions: Dict[int, Dict[str, Any]] = {}
        self._caregiver_lock = threading.Lock()

    def _prune_expired_sessions_locked(self, now: float) -> None:
        expired_ids = [
            cid
            for cid, sess in self.caregiver_sessions.items()
            if (now - float(sess.get("updated_at", 0.0))) > CAREGIVER_SESSION_TTL_SEC
        ]
        for cid in expired_ids:
            self.caregiver_sessions.pop(cid, None)

    def _get_caregiver_session(self, chat_id: int) -> Dict[str, Any]:
        now = time.monotonic()
        with self._caregiver_lock:
            self._prune_expired_sessions_locked(now)
            sess = self.caregiver_sessions.get(chat_id)
            if not sess:
                return {"enabled": False, "strips": [], "alerts": [], "updated_at": 0.0}
            return {
                "enabled": bool(sess.get("enabled", False)),
                "strips": list(sess.get("strips", [])),
                "alerts": list(sess.get("alerts", [])),
                "updated_at": float(sess.get("updated_at", 0.0)),
            }

    def _set_caregiver_enabled(self, chat_id: int, enabled: bool) -> Dict[str, Any]:
        now = time.monotonic()
        with self._caregiver_lock:
            self._prune_expired_sessions_locked(now)
            existing = self.caregiver_sessions.get(chat_id)
            strips = list(existing.get("strips", [])) if existing else []
            alerts = list(existing.get("alerts", [])) if existing else []
            self.caregiver_sessions[chat_id] = {
                "enabled": enabled,
                "strips": strips,
                "alerts": alerts,
                "updated_at": now,
            }
            return {
                "enabled": enabled,
                "strips": list(strips),
                "alerts": list(alerts),
                "updated_at": now,
            }

    def _save_caregiver_scan(
        self,
        chat_id: int,
        strips: List[StripScanResult],
        alerts: List[SafetyAlert],
    ) -> Dict[str, Any]:
        now = time.monotonic()
        with self._caregiver_lock:
            self._prune_expired_sessions_locked(now)
            existing = self.caregiver_sessions.get(chat_id)
            enabled = bool(existing.get("enabled", False)) if existing else False
            saved_strips = list(strips)
            saved_alerts = list(alerts)
            self.caregiver_sessions[chat_id] = {
                "enabled": enabled,
                "strips": saved_strips,
                "alerts": saved_alerts,
                "updated_at": now,
            }
            return {
                "enabled": enabled,
                "strips": list(saved_strips),
                "alerts": list(saved_alerts),
                "updated_at": now,
            }

    def _touch_caregiver_session(self, chat_id: int) -> None:
        now = time.monotonic()
        with self._caregiver_lock:
            sess = self.caregiver_sessions.get(chat_id)
            if sess is not None:
                sess["updated_at"] = now

    @staticmethod
    def _format_session_med_names(strips: List[StripScanResult]) -> str:
        names: List[str] = []
        for s in strips:
            m = s.matched_medicine
            if not m:
                continue
            strength = (s.extracted_strength or m.package_container or "").strip()
            names.append(f"{m.brand_name} ({strength})" if strength else m.brand_name)
        return ", ".join(names) if names else "স্ক্যানকৃত ওষুধ"

    async def api_call(self, session: aiohttp.ClientSession, method: str, **kwargs: Any) -> Dict[str, Any]:
        url = f"{API_BASE}/{method}"
        async with session.post(url, **kwargs) as resp:
            try:
                data = await resp.json(content_type=None)
            except Exception as exc:
                text = await resp.text()
                logger.error("[DEV-BOT] Non-JSON Telegram API response on %s (HTTP %d): %s", method, resp.status, text[:200])
                return {"ok": False, "error_code": resp.status, "description": str(exc)}
            if not data.get("ok"):
                logger.error("[DEV-BOT] Telegram API error on %s: %s", method, data)
            return data

    async def send_text(self, session: aiohttp.ClientSession, chat_id: int, text: str) -> Optional[int]:
        res = await self.api_call(
            session,
            "sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
            },
        )
        return res.get("result", {}).get("message_id")

    async def edit_text(
        self,
        session: aiohttp.ClientSession,
        chat_id: int,
        message_id: Optional[int],
        text: str,
    ) -> bool:
        if not message_id:
            return False
        res = await self.api_call(
            session,
            "editMessageText",
            json={
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
                "parse_mode": "HTML",
            },
        )
        return bool(res.get("ok"))

    async def delete_message(
        self,
        session: aiohttp.ClientSession,
        chat_id: int,
        message_id: Optional[int],
    ) -> None:
        if not message_id:
            return
        await self.api_call(
            session,
            "deleteMessage",
            json={"chat_id": chat_id, "message_id": message_id},
        )

    async def send_chat_action(self, session: aiohttp.ClientSession, chat_id: int, action: str = "typing") -> None:
        await self.api_call(session, "sendChatAction", json={"chat_id": chat_id, "action": action})

    async def send_voice_mp3(
        self,
        session: aiohttp.ClientSession,
        chat_id: int,
        audio_path: Path,
        caption: str,
    ) -> None:
        form = aiohttp.FormData()
        form.add_field("chat_id", str(chat_id))
        form.add_field("caption", caption[:1000])
        form.add_field("title", "OushudhCheck Bangla Voice Guide")
        form.add_field("performer", "OushudhCheck AI")
        audio_bytes = await asyncio.to_thread(audio_path.read_bytes)
        form.add_field(
            "audio",
            audio_bytes,
            filename=audio_path.name,
            content_type="audio/mpeg",
        )
        await self.api_call(session, "sendAudio", data=form)

    async def register_bot_commands(self, session: aiohttp.ClientSession) -> None:
        commands = [
            {"command": "start", "description": "স্বাগতম ও মূল মেনু (Welcome & Menu)"},
            {"command": "caregiver", "description": "কেয়ারগিভার মোড চালু/বন্ধ (Caregiver Voice Q&A)"},
            {"command": "exit", "description": "কেয়ারগিভার মোড বন্ধ করুন (Exit Caregiver Mode)"},
            {"command": "help", "description": "ব্যবহারের নিয়মাবলী (How to use)"},
            {"command": "guide", "description": "ভালো ছবি তোলার টিপস (Photo Guide)"},
            {"command": "features", "description": "AI ও নিরাপত্তা সুবিধাসমূহ (Features)"},
            {"command": "safety", "description": "চিকিৎসা সতর্কতা ও জরুরি হটলাইন (Safety)"},
            {"command": "about", "description": "প্রজেক্ট ও টিম পরিচিতি (About)"},
        ]
        await self.api_call(session, "setMyCommands", json={"commands": commands})

    async def handle_start(self, session: aiohttp.ClientSession, chat_id: int, user_name: str) -> None:
        welcome = (
            f"👋 <b>স্বাগতম {_esc(user_name)}!</b> আমি <b>OushudhCheck AI</b> (@{_esc(TELEGRAM_BOT_USERNAME)})।\n\n"
            "📸 <b>কীভাবে ব্যবহার করবেন:</b>\n"
            "১. টেবিলের ওপর ১ থেকে ৬টি ওষুধের পাতা (Medicine Strip) রেখে একটি স্পষ্ট ছবি তুলে পাঠান।\n"
            "২. 🩺 <b>কেয়ারগিভার মোড (/caregiver):</b> পাতা স্ক্যান করে বাংলা ভয়েস বা টেক্সটে ফলো-আপ প্রশ্ন করুন (যেমন: খাবারের আগে না পরে, পার্শ্বপ্রতিক্রিয়া, গর্ভাবস্থায় নিরাপদ কি না)।\n"
            "৩. 🎤 <b>বাংলা ভয়েস মেসেজ:</b> যেকোনো ওষুধের নাম বা প্রশ্ন বাংলায় ভয়েস মেসেজ পাঠিয়ে জানুন।\n"
            "৪. ⌨️ অথবা যেকোনো ওষুধের নাম লিখে পাঠান (যেমন: <code>Napa Extra</code>, <code>Seclo 20</code>, <code>Amdocal 5</code>, <code>Tufnil 200</code>)।\n\n"
            "📌 <b>মেনু কমান্ডসমূহ:</b>\n"
            "• /caregiver: কেয়ারগিভার ভয়েস প্রশ্নোত্তর মোড চালু করুন (/caregiver off বা /exit দিয়ে বন্ধ করুন)\n"
            "• /help: বিস্তারিত ব্যবহারের নিয়মাবলী\n"
            "• /guide: স্পষ্ট ছবি তোলার টিপস\n"
            "• /features: AI পাইপলাইন ও সুবিধাসমূহ\n"
            "• /safety: জরুরি চিকিৎসা সতর্কতা ও হটলাইন\n"
            "• /about: প্রজেক্ট ও টিম পরিচিতি"
        )
        await self.send_text(session, chat_id, welcome)

    async def handle_help(self, session: aiohttp.ClientSession, chat_id: int) -> None:
        msg = (
            "📘 <b>OushudhCheck ব্যবহারের নিয়মাবলী (/help)</b>\n\n"
            "<b>১. ওষুধের পাতার ছবি স্ক্যান (১–৬টি স্ট্রিপ):</b>\n"
            "• একসাথে ১ থেকে ৬টি ওষুধের পাতা সমতল টেবিলে রেখে ছবি পাঠান।\n"
            "• আমি প্রতিটি ওষুধের নাম, মাত্রা (mg), জেনেরিক, খাওয়ার নিয়ম, সহজ বাংলা সারসংক্ষেপ এবং ডুপ্লিকেট/আন্তঃক্রিয়া সতর্কতা বাংলা ভয়েসসহ জানাবো।\n\n"
            "<b>২. কেয়ারগিভার মোড (/caregiver):</b>\n"
            "• /caregiver চালু করে ওষুধের পাতার ছবি পাঠান, এরপর বাংলা ভয়েস মেসেজ বা টেক্সটে ফলো-আপ প্রশ্ন করুন।\n"
            "• উদাহরণ: “এই ওষুধটি কি খাবারের আগে না পরে খাবে?”, “পার্শ্বপ্রতিক্রিয়া কী?”, “গর্ভাবস্থায় নিরাপদ কি?”।\n"
            "• কেয়ারগিভার মোড বন্ধ করে সাধারণ ওষুধ খোঁজায় ফিরতে /caregiver off বা /exit লিখুন।\n\n"
            "<b>৩. টেক্সট লিখে খুঁজুন:</b>\n"
            "• সরাসরি ওষুধের ব্র্যান্ড বা জেনেরিক নাম লিখুন: যেমন: <code>Napa Extra</code>, <code>Seclo</code>, <code>Omeprazole</code>, <code>Amdocal 5</code>।\n\n"
            "<b>৪. বাংলা ভয়েস মেসেজ:</b>\n"
            "• মাইক্রোফোন চেপে ওষুধের নাম বলুন (যেমন: “নাপা এক্সট্রা” বা “সেকলো ২০”) অথবা স্ক্যানকৃত ওষুধ নিয়ে প্রশ্ন করুন: <b>faster-whisper</b> শুনে বাংলা ভয়েসসহ উত্তর দেবে।\n\n"
            "👉 স্পষ্ট ছবি তোলার টিপস দেখতে /guide চাপুন।"
        )
        await self.send_text(session, chat_id, msg)

    async def handle_caregiver(self, session: aiohttp.ClientSession, chat_id: int, args: str = "") -> None:
        arg_low = (args or "").strip().lower()
        if arg_low in ("off", "stop", "exit", "disable", "0"):
            self._set_caregiver_enabled(chat_id, False)
            await self.send_text(
                session,
                chat_id,
                "✅ <b>কেয়ারগিভার মোড বন্ধ করা হয়েছে।</b>\n"
                "এখন আপনি স্বাভাবিকভাবে যেকোনো ওষুধের নাম লিখে বা ছবি পাঠিয়ে তথ্য খুঁজতে পারবেন।\n"
                "আবার চালু করতে /caregiver লিখুন।",
            )
            return

        sess = self._set_caregiver_enabled(chat_id, True)
        if sess["strips"]:
            med_names = _esc(self._format_session_med_names(sess["strips"]))
            msg = (
                "🩺 <b>কেয়ারগিভার মোড চালু হয়েছে!</b>\n\n"
                f"💊 <b>বর্তমান স্ক্যানকৃত ওষুধ:</b> {med_names}\n\n"
                "🎤 এখন বাংলা <b>ভয়েস মেসেজ</b> অথবা <b>টেক্সট</b> পাঠিয়ে এই ওষুধ(গুলো) সম্পর্কে যেকোনো প্রশ্ন করুন।\n"
                "<b>উদাহরণ প্রশ্ন:</b>\n"
                "• “এই ওষুধটি কি খাবারের আগে না পরে খাবে এবং দিনে কতবার দিতে হবে?”\n"
                "• “এই ওষুধ খেলে কী কী পার্শ্বপ্রতিক্রিয়া হতে পারে?”\n"
                "• “গর্ভাবস্থায় বা স্তন্যদানকালে কি এই ওষুধটি খাওয়া নিরাপদ?”\n\n"
                "📸 নতুন ওষুধের পাতার ছবি পাঠালে সেটিও স্বয়ংক্রিয়ভাবে আপডেট হয়ে যাবে।\n"
                "ℹ️ সাধারণ ওষুধ খোঁজায় ফিরতে /caregiver off বা /exit লিখুন।"
            )
        else:
            msg = (
                "🩺 <b>কেয়ারগিভার মোড চালু হয়েছে!</b>\n\n"
                "📸 <b>ধাপ ১:</b> প্রথমে ওষুধের পাতার (১–৬টি স্ট্রিপ) একটি স্পষ্ট ছবি পাঠান।\n"
                "🎤 <b>ধাপ ২:</b> স্ক্যান সম্পন্ন হলে বাংলা <b>ভয়েস মেসেজ</b> বা <b>টেক্সটে</b> ফলো-আপ প্রশ্ন করুন।\n\n"
                "<b>উদাহরণ প্রশ্ন:</b>\n"
                "• “এই ওষুধটি কি খাবারের আগে না পরে খাবে?”\n"
                "• “এই ওষুধ খেলে কী কী পার্শ্বপ্রতিক্রিয়া হতে পারে?”\n"
                "• “অন্য কোনো ওষুধের সাথে খেলে কি কোনো সমস্যা আছে?”\n\n"
                "ℹ️ সাধারণ ওষুধ খোঁজায় ফিরতে /caregiver off বা /exit লিখুন।"
            )
        await self.send_text(session, chat_id, msg)

    async def handle_exit(self, session: aiohttp.ClientSession, chat_id: int) -> None:
        sess = self._get_caregiver_session(chat_id)
        if sess["enabled"]:
            self._set_caregiver_enabled(chat_id, False)
            await self.send_text(
                session,
                chat_id,
                "✅ <b>কেয়ারগিভার মোড বন্ধ করা হয়েছে।</b>\n"
                "এখন আপনি স্বাভাবিকভাবে ওষুধের নাম লিখে বা ছবি পাঠিয়ে খুঁজতে পারবেন।",
            )
        else:
            await self.send_text(
                session,
                chat_id,
                "ℹ️ কেয়ারগিভার মোড বর্তমানে চালু নেই। চালু করতে /caregiver লিখুন।",
            )

    async def handle_guide(self, session: aiohttp.ClientSession, chat_id: int) -> None:
        msg = (
            "📷 <b>ভালো ছবি তোলার টিপস (/guide)</b>\n\n"
            "✅ <b>কী করবেন:</b>\n"
            "• ওষুধের পাতার <b>লেখার দিকটি</b> (যেখানে ব্র্যান্ড নাম ও mg লেখা আছে) ক্যামেরার দিকে রাখুন।\n"
            "• পর্যাপ্ত আলোতে ছবি তুলুন, তবে সরাসরি ফ্ল্যাশ বা চকচকে ফয়েলের ঝলকানি এড়িয়ে চলুন।\n"
            "• একাধিক পাতা একসাথে রাখলে একটু ফাঁকা রেখে সাজান যাতে নাম ঢেকে না যায়।\n"
            "• ছবি তোলার আগে ক্যামেরা ফোকাস করে নিন যাতে লেখা ঝাপসা না হয়।\n\n"
            "❌ <b>কী এড়িয়ে চলবেন:</b>\n"
            "• খুব অন্ধকার পরিবেশ, অতিরিক্ত ঝাপসা ছবি বা উল্টো/ক্যাপসুলের দিকের ছবি।"
        )
        await self.send_text(session, chat_id, msg)

    async def handle_features(self, session: aiohttp.ClientSession, chat_id: int) -> None:
        msg = (
            "✨ <b>OushudhCheck-এর AI সুবিধাসমূহ (/features)</b>\n\n"
            "• 🔍 <b>মাল্টি-স্ট্রিপ ডিটেকশন (YOLO26s ONNX):</b> এক ছবিতেই ১–৬টি ওষুধের পাতা শনাক্ত করে।\n"
            "• 🧾 <b>ফয়েল টেক্সট রিডার (PP-OCRv6 + CLAHE):</b> চকচকে ফয়েল ও বাঁকা/লম্বালম্বি লেখা পড়ে।\n"
            "• 💊 <b>১৭,১৯১+ বাংলাদেশি ওষুধের ডাটাবেস:</b> ১,৬৬১+ জেনেরিক ও ব্র্যান্ড ম্যাচিং (RapidFuzz)।\n"
            "• 🚨 <b>ডুপ্লিকেট ও ওভারডোজ গার্ড:</b> ভিন্ন ব্র্যান্ড কিন্তু একই জেনেরিক (যেমন Napa + Ace = Paracetamol) শনাক্ত করে সতর্ক করে।\n"
            "• ⚠️ <b>Drug-Drug Interaction (DDI) স্ক্রিনিং:</b> একসাথে খাওয়ার ঝুঁকিপূর্ণ ওষুধ যাচাই করে।\n"
            "• 🤖 <b>Gemma 3 1B বাংলা SLM:</b> সহজ বাংলায় ওষুধের কাজ, খাওয়ার নিয়ম ও সতর্কতা তৈরি করে।\n"
            "• 🔊 <b>বাংলা নিউরাল ভয়েস (Edge TTS):</b> পড়তে অসুবিধা হলে পুরো রিপোর্ট বাংলায় শুনুন।"
        )
        await self.send_text(session, chat_id, msg)

    async def handle_safety(self, session: aiohttp.ClientSession, chat_id: int) -> None:
        msg = (
            "🛡️ <b>জরুরি চিকিৎসা সতর্কতা ও হটলাইন (/safety)</b>\n\n"
            "⚠️ <b>মেডিকেল ডিসক্লেইমার:</b>\n"
            "OushudhCheck একটি AI-সহায়তাকৃত ওষুধ তথ্য ও প্রাথমিক নিরাপত্তা যাচাই টুল। এটি কোনো চিকিৎসকের প্রেসক্রিপশন বা রোগ নির্ণয়ের বিকল্প নয়। "
            "যেকোনো ওষুধ শুরু, পরিবর্তন বা বন্ধ করার আগে অবশ্যই রেজিস্টার্ড চিকিৎসকের পরামর্শ নিন।\n\n"
            "📞 <b>বাংলাদেশ জরুরি স্বাস্থ্য হটলাইন:</b>\n"
            "• স্বাস্থ্য বাতায়ন (২৪/৭ ডাক্তার পরামর্শ): <code>16263</code>\n"
            "• জাতীয় জরুরি সেবা: <code>999</code>"
        )
        await self.send_text(session, chat_id, msg)

    async def handle_about(self, session: aiohttp.ClientSession, chat_id: int) -> None:
        msg = (
            "🏥 <b>OushudhCheck: Know Your Medicines. Protect Your Health.</b>\n\n"
            "✨ <b>AI-Powered Medicine Safety &amp; Bangla Voice Assistant</b>\n\n"
            "👨‍💻 <b>Team OushudhCheck:</b>\n"
            "• <b>Jubayer Ahmed</b>: Frontend &amp; AI Product Engineer\n"
            "• <b>Ridoy Hossan Rony</b>: Backend &amp; ML Systems Engineer\n\n"
            "💡 আমাদের লক্ষ্য: বাংলাদেশে সবার জন্য সহজ বাংলায় নিরাপদ ওষুধ তথ্য ও ভয়েস গাইড নিশ্চিত করা।\n"
            "সাহায্যের জন্য /help বা /guide লিখুন।"
        )
        await self.send_text(session, chat_id, msg)

    async def _download_bounded(self, session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
        async with session.get(url) as dl_resp:
            if dl_resp.status != 200:
                return None
            chunks: List[bytes] = []
            total = 0
            async for chunk in dl_resp.content.iter_chunked(64 * 1024):
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    return None
                chunks.append(chunk)
            return b"".join(chunks)

    async def _build_guidance_parallel(
        self,
        matched_strips: List[Any],
        dup_ids: set,
        inter_ids: set,
    ) -> List[Any]:
        """Run per-strip Gemma 3 1B guidance generation off the event loop in parallel (bounded to 2 slots)."""
        if not matched_strips:
            return []
        sem = asyncio.Semaphore(2)

        async def _one(strip: Any) -> Any:
            async with sem:
                try:
                    return await asyncio.to_thread(
                        self.scanner.slm.build_schedule_and_guidance,
                        strip,
                        is_duplicate=(strip.strip_index in dup_ids),
                        is_interaction=(strip.strip_index in inter_ids),
                    )
                except Exception as exc:
                    logger.exception("[DEV-BOT] SLM guidance failed for strip %s: %s", strip.strip_index, exc)
                    med = strip.matched_medicine
                    brand = med.brand_name if med else "ওষুধ"
                    return (
                        "চিকিৎসকের পরামর্শ অনুযায়ী সেব্য",
                        f"{brand} চিকিৎসকের পরামর্শ অনুযায়ী সেবন করুন।",
                        "চিকিৎসকের পরামর্শ অনুযায়ী",
                    )

        return list(await asyncio.gather(*(_one(s) for s in matched_strips)))

    async def handle_photo(
        self,
        session: aiohttp.ClientSession,
        chat_id: int,
        file_id: str,
        user_name: str,
    ) -> None:
        t0 = time.perf_counter()
        logger.info("[DEV-BOT] Received photo from %s (chat_id=%d, file_id=%s)", user_name, chat_id, file_id)

        # Stage 1/5 Progress
        progress_msg_id = await self.send_text(
            session,
            chat_id,
            "<b>[ধাপ ১/৫]</b> ছবি ডাউনলোড ও <b>YOLO26 ONNX</b> স্ট্রিপ ডিটেকশন চলছে...\n"
            "<code>[■■□□□□□□□□] 20%</code>",
        )
        await self.send_chat_action(session, chat_id, "typing")

        file_info = await self.api_call(session, "getFile", json={"file_id": file_id})
        file_path = file_info.get("result", {}).get("file_path")
        if not file_path:
            await self.edit_text(session, chat_id, progress_msg_id, "❌ ছবিটি ডাউনলোড করা যায়নি। অনুগ্রহ করে আবার চেষ্টা করুন।")
            return

        img_bytes = await self._download_bounded(session, f"{FILE_BASE}/{file_path}")
        if not img_bytes:
            await self.edit_text(session, chat_id, progress_msg_id, "❌ ছবিটি ডাউনলোড করা যায়নি অথবা ফাইলের আকার ১৫ MB এর বেশি।")
            return

        def _decode_img(raw: bytes) -> Optional[np.ndarray]:
            arr = np.frombuffer(raw, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

        image = await asyncio.to_thread(_decode_img, img_bytes)
        if image is None:
            await self.edit_text(session, chat_id, progress_msg_id, "❌ ছবির ফরম্যাট পড়া যায়নি। স্পষ্ট JPEG/PNG ছবি পাঠান।")
            return

        h, w = image.shape[:2]
        if h * w > 25_000_000:
            await self.edit_text(session, chat_id, progress_msg_id, "❌ ছবির রেজোলিউশন অনেক বড় (সর্বোচ্চ ২৫ মেগাপিক্সেল)।")
            return

        detection, yolo_ms = await asyncio.to_thread(self.scanner.run_stage1_detect, image)
        yolo_note = (
            f" ({detection.raw_count}টি স্ট্রিপ শনাক্ত • {yolo_ms}ms)"
            if detection.is_real_model
            else ""
        )

        # Stage 2/5 Progress
        await self.edit_text(
            session,
            chat_id,
            progress_msg_id,
            f"<b>[ধাপ ২/৫]</b> YOLO26{yolo_note} সম্পন্ন। <b>CLAHE + PP-OCRv6</b> টেক্সট এক্সট্রাকশন চলছে...\n"
            "<code>[■■■■□□□□□□] 45%</code>",
        )
        ocr_lines, _, _ = await asyncio.to_thread(self.scanner.run_stage2_3_ocr, image)

        # Stage 3/5 Progress
        await self.edit_text(
            session,
            chat_id,
            progress_msg_id,
            f"<b>[ধাপ ৩/৫]</b> {len(ocr_lines)}টি টেক্সট লাইন পাওয়া গেছে। <b>YOLO+OCR ফিউশন ও ১৭,১৯১টি ওষুধের ডাটাবেস</b> ম্যাচিং চলছে...\n"
            "<code>[■■■■■■■□□□] 70%</code>",
        )

        strip_results, _, _, _ = await asyncio.to_thread(
            self.scanner.run_stage4_fuse_and_match, image, detection, ocr_lines
        )

        matched = [s for s in strip_results if s.matched_medicine is not None]
        if not matched:
            await self.edit_text(
                session,
                chat_id,
                progress_msg_id,
                "⚠️ ছবি থেকে কোনো ওষুধের নাম স্পষ্টভাবে পড়া যায়নি। অনুগ্রহ করে পর্যাপ্ত আলোতে ওষুধের পাতার লেখার দিকটি সামনে রেখে ছবি তুলুন।",
            )
            return

        # Stage 4/5 Progress
        await self.edit_text(
            session,
            chat_id,
            progress_msg_id,
            f"<b>[ধাপ ৪/৫]</b> {len(matched)}টি ওষুধ শনাক্ত হয়েছে! <b>Gemma 3 1B Bangla SLM</b> বাংলা সারসংক্ষেপ ও ডুপ্লিকেট/আন্তঃক্রিয়া যাচাই চলছে...\n"
            "<code>[■■■■■■■■■□] 90%</code>",
        )
        alerts, _, _ = evaluate_multi_strip_safety(strip_results)
        caregiver_sess = self._save_caregiver_scan(chat_id, matched, alerts)

        dup_ids = {sid for a in alerts if a.alert_type == "duplicate_generic" for sid in a.involved_strips}
        inter_ids = {sid for a in alerts if a.alert_type != "duplicate_generic" for sid in a.involved_strips}

        guidance_list = await self._build_guidance_parallel(matched, dup_ids, inter_ids)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        lines_out = [
            f"💊 <b>OushudhCheck স্ক্যান রিপোর্ট</b> (<i>{len(matched)}টি স্ট্রিপ • {elapsed_ms}ms</i>)\n"
        ]
        voice_parts = [f"আপনার ছবিতে {len(matched)}টি ওষুধের পাতা শনাক্ত করা হয়েছে।"]

        for idx, (s, (dosage_bn, bangla_summary, timing_tag)) in enumerate(zip(matched, guidance_list), start=1):
            m = s.matched_medicine
            assert m is not None
            strength = s.extracted_strength or m.package_container or "N/A"
            form = s.detected_dosage_form or m.dosage_form or "Tablet"
            lines_out.append(
                f"<b>{idx}. {_esc(m.brand_name)}</b>: <code>{_esc(strength)}</code> ({_esc(form)})\n"
                f"   • <b>জেনেরিক:</b> {_esc(m.generic)}\n"
                f"   • <b>প্রস্তুতকারক:</b> {_esc(m.manufacturer or 'N/A')}\n"
                f"   • <b>খাওয়ার সময়সূচী:</b> <code>{_esc(timing_tag)}</code>\n"
                f"   • <b>নির্দেশনা:</b> {_esc(dosage_bn)}\n"
                f"   • <b>বাংলা সারসংক্ষেপ:</b> {_format_tg_summary(bangla_summary, indent='   ')}\n"
            )
            voice_parts.append(f"স্ট্রিপ {idx}: {bangla_summary}")

        if alerts:
            lines_out.append("\n🚨 <b>সতর্কতা (Safety Alerts):</b>")
            for a in alerts:
                lines_out.append(f"• ⚠️ <b>{_esc(a.message_bn)}</b>")
                voice_parts.append(a.message_bn)
        else:
            lines_out.append("✅ <b>নিরাপত্তা যাচাই:</b> শনাক্তকৃত ওষুধে কোনো ডুপ্লিকেট জেনেরিক বা ঝুঁকিপূর্ণ আন্তঃক্রিয়া পাওয়া যায়নি।")
            voice_parts.append("ওষুধগুলোতে কোনো ডুপ্লিকেট উপাদান পাওয়া যায়নি।")

        if caregiver_sess["enabled"]:
            lines_out.append(
                "\n🩺 <b>কেয়ারগিভার মোড চালু আছে:</b> এই ওষুধ(গুলো) নিয়ে এখন বাংলা <b>ভয়েস মেসেজ</b> বা <b>টেক্সটে</b> ফলো-আপ প্রশ্ন করুন।\n"
                "• “এই ওষুধটি কি খাবারের আগে না পরে খাবে?”\n"
                "• “এর পার্শ্বপ্রতিক্রিয়া কী কী?”\n"
                "• “গর্ভাবস্থায় বা স্তন্যদানকালে নিরাপদ কি?”\n"
                "<i>(বন্ধ করতে /caregiver off বা /exit লিখুন)</i>"
            )
        else:
            lines_out.append(
                "\n🎤 <b>ফলো-আপ প্রশ্ন করতে চান?</b> এই ওষুধ(গুলো) নিয়ে সরাসরি বাংলা ভয়েস মেসেজে প্রশ্ন করুন (৩০ মিনিট পর্যন্ত মনে থাকবে), অথবা /caregiver চালু করুন:\n"
                "• “খাবারের আগে না পরে খাবে?”\n"
                "• “কী কী পার্শ্বপ্রতিক্রিয়া হতে পারে?”\n"
                "• “অন্য ওষুধের সাথে খেলে কি সমস্যা আছে?”"
            )

        # Remove the temporary progress bar
        await self.delete_message(session, chat_id, progress_msg_id)
        report_chunks = _chunk_html_blocks(lines_out)
        for chunk in report_chunks:
            await self.send_text(session, chat_id, chunk)

        # Stage 5/5 Voice Generation: "Voice is generating..." status message
        voice_status_id = await self.send_text(
            session,
            chat_id,
            "🔊 <i>বাংলা ভয়েস অডিও তৈরি হচ্ছে... অনুগ্রহ করে অপেক্ষা করুন (Generating Bangla Voice Audio...)</i>",
        )
        await self.send_chat_action(session, chat_id, "record_voice")
        voice_script = " ".join(voice_parts)
        audio_path = await synthesize_bangla_speech(voice_script)
        if audio_path and audio_path.exists():
            await self.send_voice_mp3(
                session,
                chat_id,
                audio_path,
                caption="🔊 বাংলা ভয়েস গাইড (OushudhCheck AI)",
            )
            await self.delete_message(session, chat_id, voice_status_id)
        else:
            await self.edit_text(
                session,
                chat_id,
                voice_status_id,
                "⚠️ ভয়েস অডিও তৈরি করা সম্ভব হয়নি, তবে উপরের লিখিত রিপোর্টটি সম্পূর্ণ যাচাইকৃত।",
            )

    async def handle_caregiver_followup(
        self,
        session: aiohttp.ClientSession,
        chat_id: int,
        question: str,
        user_name: str,
        existing_msg_id: Optional[int] = None,
        via_voice: bool = False,
    ) -> None:
        logger.info(
            "[DEV-BOT] Caregiver follow-up from %s (chat_id=%d, voice=%s): %r",
            user_name,
            chat_id,
            via_voice,
            question,
        )
        sess = self._get_caregiver_session(chat_id)
        session_strips: List[StripScanResult] = sess["strips"]
        session_alerts: List[SafetyAlert] = sess["alerts"]

        if not session_strips:
            no_scan_msg = (
                "📸 <b>প্রথমে ওষুধের পাতার ছবি পাঠান!</b>\n\n"
                "কেয়ারগিভার মোডে ফলো-আপ প্রশ্নের উত্তর দেওয়ার জন্য আগে ১ থেকে ৬টি ওষুধের পাতার স্পষ্ট ছবি স্ক্যান করুন।\n"
                "<i>(সাধারণ ওষুধ খোঁজায় ফিরতে /caregiver off বা /exit লিখুন)</i>"
            )
            if existing_msg_id:
                await self.edit_text(session, chat_id, existing_msg_id, no_scan_msg)
            else:
                await self.send_text(session, chat_id, no_scan_msg)
            return

        self._touch_caregiver_session(chat_id)
        status_text = (
            f"🗣️ <b>আপনি জিজ্ঞাসা করেছেন:</b> <i>“{_esc(question)}”</i>\n"
            "🤖 <b>OushudhCheck AI</b> স্ক্যানকৃত ওষুধের তথ্য দেখে বাংলা উত্তর তৈরি করছে..."
        )
        if existing_msg_id:
            await self.edit_text(session, chat_id, existing_msg_id, status_text)
        else:
            existing_msg_id = await self.send_text(session, chat_id, status_text)
        await self.send_chat_action(session, chat_id, "typing")

        try:
            answer_bn = await asyncio.to_thread(
                answer_caregiver_followup,
                question,
                session_strips,
                session_alerts,
            )
        except Exception as exc:
            logger.exception("[DEV-BOT] Caregiver follow-up failed for chat_id=%d: %s", chat_id, exc)
            answer_bn = (
                "দুঃখিত, এই মুহূর্তে উত্তর তৈরি করা যায়নি। অনুগ্রহ করে প্রশ্নটি আবার করুন অথবা "
                "যেকোনো চিকিৎসা সিদ্ধান্তের জন্য রেজিস্টার্ড চিকিৎসকের পরামর্শ নিন।"
            )

        med_names = _esc(self._format_session_med_names(session_strips))
        footer = (
            "💡 <i>আরও প্রশ্ন থাকলে বাংলা ভয়েস বা টেক্সট পাঠান। সাধারণ ওষুধ খোঁজায় ফিরতে /caregiver off লিখুন।</i>"
            if sess["enabled"]
            else "💡 <i>আরও প্রশ্ন থাকলে বাংলা ভয়েস বা টেক্সটে জিজ্ঞাসা করুন, অথবা /caregiver চালু করুন।</i>"
        )
        blocks = [
            f"🩺 <b>কেয়ারগিভার উত্তর (Caregiver Q&amp;A)</b>\n💊 <b>ওষুধ:</b> {med_names}",
            f"🗣️ <b>আপনি জিজ্ঞাসা করেছেন:</b> <i>“{_esc(question)}”</i>",
            f"📖 <b>উত্তর:</b>\n{_format_tg_summary(answer_bn)}",
            footer,
        ]

        await self.delete_message(session, chat_id, existing_msg_id)
        for chunk in _chunk_html_blocks(blocks):
            await self.send_text(session, chat_id, chunk)

        voice_status_id = await self.send_text(
            session,
            chat_id,
            "🔊 <i>বাংলা ভয়েস উত্তর তৈরি হচ্ছে... (Generating Bangla Voice Reply...)</i>",
        )
        await self.send_chat_action(session, chat_id, "record_voice")
        audio_path = await synthesize_bangla_speech(answer_bn)
        if audio_path and audio_path.exists():
            await self.send_voice_mp3(
                session,
                chat_id,
                audio_path,
                caption="🔊 কেয়ারগিভার বাংলা ভয়েস উত্তর (OushudhCheck AI)",
            )
            await self.delete_message(session, chat_id, voice_status_id)
        else:
            await self.edit_text(
                session,
                chat_id,
                voice_status_id,
                "⚠️ ভয়েস উত্তর তৈরি করা সম্ভব হয়নি, তবে উপরের লিখিত উত্তরটি সম্পূর্ণ যাচাইকৃত।",
            )

    async def handle_voice_message(
        self,
        session: aiohttp.ClientSession,
        chat_id: int,
        file_id: str,
        user_name: str,
    ) -> None:
        logger.info("[DEV-BOT] Received voice message from %s (chat_id=%d, file_id=%s)", user_name, chat_id, file_id)
        pre_sess = self._get_caregiver_session(chat_id)
        if pre_sess["enabled"] and not pre_sess["strips"]:
            await self.send_text(
                session,
                chat_id,
                "📸 <b>প্রথমে ওষুধের পাতার ছবি পাঠান!</b>\n\n"
                "কেয়ারগিভার মোডে ভয়েস ফলো-আপ প্রশ্নের উত্তর দেওয়ার জন্য আগে ১ থেকে ৬টি ওষুধের পাতার স্পষ্ট ছবি স্ক্যান করুন।\n"
                "<i>(সাধারণ ওষুধ খোঁজায় ফিরতে /caregiver off বা /exit লিখুন)</i>",
            )
            return

        p_id = await self.send_text(
            session,
            chat_id,
            "🎤 <b>[ধাপ ১/৩]</b> আপনার ভয়েস মেসেজটি শোনা হচ্ছে (<b>faster-whisper</b>)...\n<code>[■■■□□□□□□□] 33%</code>",
        )

        file_info = await self.api_call(session, "getFile", json={"file_id": file_id})
        file_path = file_info.get("result", {}).get("file_path")
        if not file_path:
            await self.edit_text(session, chat_id, p_id, "❌ ভয়েস ফাইলটি ডাউনলোড করা যায়নি।")
            return

        voice_bytes = await self._download_bounded(session, f"{FILE_BASE}/{file_path}")
        if not voice_bytes:
            await self.edit_text(session, chat_id, p_id, "❌ ভয়েস ফাইলটি ডাউনলোড করা যায়নি।")
            return

        tmp_ogg = Path(tempfile.gettempdir()) / f"tg_voice_{chat_id}_{uuid.uuid4().hex}.ogg"
        await asyncio.to_thread(tmp_ogg.write_bytes, voice_bytes)

        try:
            transcribed = await asyncio.to_thread(transcribe_bangla_audio, tmp_ogg)
        finally:
            tmp_ogg.unlink(missing_ok=True)

        if not transcribed:
            await self.edit_text(
                session,
                chat_id,
                p_id,
                "⚠️ আপনার ভয়েস মেসেজটি স্পষ্টভাবে বোঝা যায়নি। অনুগ্রহ করে আবার বলুন অথবা ওষুধের নাম লিখে পাঠান।",
            )
            return

        sess = self._get_caregiver_session(chat_id)
        if sess["enabled"] or (sess["strips"] and _looks_like_followup_question(transcribed)):
            await self.handle_caregiver_followup(
                session,
                chat_id,
                transcribed,
                user_name,
                existing_msg_id=p_id,
                via_voice=True,
            )
            return

        await self.edit_text(
            session,
            chat_id,
            p_id,
            f"🗣️ <b>আপনি বলেছেন:</b> <i>“{_esc(transcribed)}”</i>\n🔍 ডাটাবেস থেকে তথ্য খোঁজা হচ্ছে...",
        )
        await self.handle_text_query(session, chat_id, transcribed, user_name, existing_msg_id=p_id)

    def _split_multi_medicine_query(self, text: str) -> List[str]:
        """
        Splits multi-medicine user messages like:
        - 'napa 500mg vs ace 500mg'
        - 'napa 500mg ace 500mg'
        - 'napa 500mg, seclo 20mg and ace'
        - 'napa ও ace'
        into separate per-medicine query chunks so ALL medicines are matched and checked for duplicates/interactions.
        """
        # 1. Split on explicit delimiters (vs, and, with, comma, semicolon, newline, Bangla connectors)
        explicit_parts = re.split(
            r"(?:[,;\n&/]+|\s+(?:vs\.?|versus|and|with|or|এবং|ও|সাথে|বনাম)\s+|\s+\+\s+(?=[A-Za-z]))",
            text.strip(),
            flags=re.IGNORECASE,
        )
        chunks: List[str] = []
        for part in explicit_parts:
            p = part.strip()
            if not p:
                continue
            # 2. Split space-separated name
            sub_parts = re.sub(
                r"(\d\s*(?:mcg|mg|ml|iu|gm|g))\s+(?=[A-Za-z])",
                r"\1\n",
                p,
                flags=re.IGNORECASE,
            ).splitlines()
            for sp in sub_parts:
                sp_clean = sp.strip()
                if not sp_clean:
                    continue
                # 3. Handle bare space-separated brands like 'napa ace seclo' (when >=2 known brands appear)
                words = sp_clean.split()
                if len(words) >= 2 and not re.search(r"\d", sp_clean):
                    known_brands = self.scanner.matcher._brands_by_lower
                    if sp_clean.lower() not in known_brands:
                        matched_tokens = [w for w in words if w.lower() in known_brands]
                        if len(matched_tokens) >= 2:
                            chunks.extend(matched_tokens)
                            continue
                chunks.append(sp_clean)
        return chunks[:6] or [text.strip()]

    async def handle_text_query(
        self,
        session: aiohttp.ClientSession,
        chat_id: int,
        text: str,
        user_name: str,
        existing_msg_id: Optional[int] = None,
    ) -> None:
        logger.info("[DEV-BOT] Received text query from %s: %r", user_name, text)
        if not existing_msg_id:
            existing_msg_id = await self.send_text(
                session,
                chat_id,
                f"🔍 <code>{_esc(text)}</code> ওষুধের তথ্য খোঁজা হচ্ছে (<b>RapidFuzz + Gemma 3 1B SLM</b>)...",
            )

        query_chunks = self._split_multi_medicine_query(text)

        def _match_all_queries():
            matched_list = []
            seen_keys = set()
            for idx, q_part in enumerate(query_chunks, start=1):
                res = self.scanner.matcher.match_strip_text(
                    q_part,
                    None,
                    idx,
                    None,
                    None,
                    100.0,
                    "exact_brand",
                    self.scanner.slm,
                )
                if res.matched_medicine is not None:
                    key = (res.matched_medicine.brand_name.lower(), (res.extracted_strength or "").lower())
                    if key not in seen_keys:
                        seen_keys.add(key)
                        res.strip_index = len(matched_list) + 1
                        matched_list.append(res)
            return matched_list

        matched_results = await asyncio.to_thread(_match_all_queries)
        if not matched_results:
            await self.edit_text(
                session,
                chat_id,
                existing_msg_id,
                f"🔍 <code>{_esc(text)}</code> নামে কোনো ওষুধ ডাটাবেসে পাওয়া যায়নি। সঠিক বানান লিখুন অথবা ওষুধের পাতার ছবি পাঠান।",
            )
            return

        # Single-medicine reply
        if len(matched_results) == 1:
            res = matched_results[0]
            m = res.matched_medicine
            assert m is not None
            strength = res.extracted_strength or m.package_container or "Standard"
            form = res.detected_dosage_form or m.dosage_form or "Tablet"
            dosage_bn, bangla_summary, timing_tag = await asyncio.to_thread(
                self.scanner.slm.build_schedule_and_guidance,
                res,
                False,
                False,
            )
            reply = (
                f"💊 <b>{_esc(m.brand_name)}</b>: <code>{_esc(strength)}</code> ({_esc(form)})\n"
                f"• <b>জেনেরিক:</b> {_esc(m.generic)}\n"
                f"• <b>শ্রেণী:</b> {_esc(m.drug_class or 'N/A')}\n"
                f"• <b>প্রস্তুতকারক:</b> {_esc(m.manufacturer or 'N/A')}\n"
                f"• <b>খাওয়ার সময়সূচী:</b> <code>{_esc(timing_tag)}</code>\n"
                f"• <b>নির্দেশনা:</b> {_esc(dosage_bn)}\n\n"
                f"📖 <b>সহজ বাংলা সারসংক্ষেপ:</b>\n{_format_tg_summary(bangla_summary)}"
            )
            await self.delete_message(session, chat_id, existing_msg_id)
            for chunk in _chunk_html_blocks(reply.split("\n\n")):
                await self.send_text(session, chat_id, chunk)

            voice_status_id = await self.send_text(
                session,
                chat_id,
                "🔊 <i>বাংলা ভয়েস অডিও তৈরি হচ্ছে... (Generating Bangla Voice...)</i>",
            )
            await self.send_chat_action(session, chat_id, "record_voice")
            audio_path = await synthesize_bangla_speech(bangla_summary)
            if audio_path and audio_path.exists():
                await self.send_voice_mp3(
                    session,
                    chat_id,
                    audio_path,
                    caption=f"🔊 {m.brand_name}: বাংলা ভয়েস গাইড (OushudhCheck AI)",
                )
                await self.delete_message(session, chat_id, voice_status_id)
            else:
                await self.edit_text(
                    session, chat_id, voice_status_id,
                    "⚠️ ভয়েস অডিও তৈরি করা সম্ভব হয়নি, তবে লিখিত রিপোর্টটি সম্পূর্ণ যাচাইকৃত।"
                )
            return

        # Multi-medicine comparison & safety check reply (e.g., 'napa 500mg vs ace 500mg')
        alerts, _, _ = evaluate_multi_strip_safety(matched_results)
        dup_ids = {sid for a in alerts if a.alert_type == "duplicate_generic" for sid in a.involved_strips}
        inter_ids = {sid for a in alerts if a.alert_type != "duplicate_generic" for sid in a.involved_strips}

        guidance_list = await self._build_guidance_parallel(matched_results, dup_ids, inter_ids)
        lines_out = [f"💊 <b>OushudhCheck তুলনা ও নিরাপত্তা যাচাই</b> (<i>{len(matched_results)}টি ওষুধ</i>)\n"]
        voice_parts = [f"আপনার প্রশ্নে {len(matched_results)}টি ওষুধ পাওয়া গেছে।"]

        for idx, (r, (dosage_bn, bangla_summary, timing_tag)) in enumerate(zip(matched_results, guidance_list), start=1):
            m = r.matched_medicine
            assert m is not None
            strength = r.extracted_strength or m.package_container or "Standard"
            form = r.detected_dosage_form or m.dosage_form or "Tablet"
            lines_out.append(
                f"<b>{idx}. {_esc(m.brand_name)}</b>: <code>{_esc(strength)}</code> ({_esc(form)})\n"
                f"   • <b>জেনেরিক:</b> {_esc(m.generic)}\n"
                f"   • <b>প্রস্তুতকারক:</b> {_esc(m.manufacturer or 'N/A')}\n"
                f"   • <b>খাওয়ার সময়সূচী:</b> <code>{_esc(timing_tag)}</code>\n"
                f"   • <b>নির্দেশনা:</b> {_esc(dosage_bn)}\n"
                f"   • <b>বাংলা সারসংক্ষেপ:</b> {_format_tg_summary(bangla_summary, indent='   ')}\n"
            )
            voice_parts.append(f"ওষুধ {idx}: {bangla_summary}")

        if alerts:
            lines_out.append("\n🚨 <b>সতর্কতা (Safety Alerts):</b>")
            for a in alerts:
                lines_out.append(f"• ⚠️ <b>{_esc(a.message_bn)}</b>")
                voice_parts.append(a.message_bn)
        else:
            lines_out.append("✅ <b>নিরাপত্তা যাচাই:</b> এই ওষুধগুলোতে কোনো ডুপ্লিকেট জেনেরিক বা ঝুঁকিপূর্ণ আন্তঃক্রিয়া পাওয়া যায়নি।")
            voice_parts.append("এই ওষুধগুলোতে কোনো ডুপ্লিকেট উপাদান বা ঝুঁকিপূর্ণ আন্তঃক্রিয়া পাওয়া যায়নি।")

        await self.delete_message(session, chat_id, existing_msg_id)
        for chunk in _chunk_html_blocks(lines_out):
            await self.send_text(session, chat_id, chunk)

        voice_status_id = await self.send_text(
            session,
            chat_id,
            "🔊 <i>বাংলা ভয়েস অডিও তৈরি হচ্ছে... (Generating Bangla Voice...)</i>",
        )
        await self.send_chat_action(session, chat_id, "record_voice")
        audio_path = await synthesize_bangla_speech(" ".join(voice_parts))
        if audio_path and audio_path.exists():
            await self.send_voice_mp3(session, chat_id, audio_path, caption="🔊 বাংলা ভয়েস গাইড (OushudhCheck AI)")
        await self.delete_message(session, chat_id, voice_status_id)

    async def poll_forever(self) -> None:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            me = await self.api_call(session, "getMe")
            if me.get("ok"):
                bot_info = me["result"]
                logger.info(
                    "[DEV-BOT] Connected to Telegram Bot @%s (id=%s, name=%r)",
                    bot_info.get("username"),
                    bot_info.get("id"),
                    bot_info.get("first_name"),
                )
                await self.register_bot_commands(session)
            else:
                logger.error("[DEV-BOT] Failed to authenticate Telegram bot token!")
                return

            sem = asyncio.Semaphore(4)
            active_tasks: set = set()

            async def _dispatch_update(msg: Dict[str, Any]) -> None:
                async with sem:
                    progress_msg_id: Optional[int] = None
                    try:
                        chat_id = int(msg["chat"]["id"])
                        user_name = msg.get("from", {}).get("first_name", "User")

                        if "photo" in msg and msg["photo"]:
                            best_photo = msg["photo"][-1]
                            await self.handle_photo(session, chat_id, best_photo["file_id"], user_name)
                        elif "voice" in msg and msg["voice"]:
                            await self.handle_voice_message(session, chat_id, msg["voice"]["file_id"], user_name)
                        elif "text" in msg:
                            txt = str(msg["text"]).strip()
                            cmd = txt.split()[0].split("@")[0].lower() if txt.startswith("/") else ""
                            cmd_args = txt.split(maxsplit=1)[1].strip() if len(txt.split(maxsplit=1)) > 1 else ""
                            if cmd == "/start":
                                await self.handle_start(session, chat_id, user_name)
                            elif cmd == "/caregiver":
                                await self.handle_caregiver(session, chat_id, cmd_args)
                            elif cmd == "/exit":
                                await self.handle_exit(session, chat_id)
                            elif cmd == "/help":
                                await self.handle_help(session, chat_id)
                            elif cmd == "/guide":
                                await self.handle_guide(session, chat_id)
                            elif cmd == "/features":
                                await self.handle_features(session, chat_id)
                            elif cmd == "/safety":
                                await self.handle_safety(session, chat_id)
                            elif cmd == "/about":
                                await self.handle_about(session, chat_id)
                            elif cmd:
                                await self.send_text(
                                    session,
                                    chat_id,
                                    "❓ অজানা কমান্ড। ব্যবহারের নিয়ম দেখতে /help অথবা মূল মেনু দেখতে /start চাপুন।",
                                )
                            else:
                                sess = self._get_caregiver_session(chat_id)
                                if sess["enabled"] or (sess["strips"] and _looks_like_followup_question(txt)):
                                    await self.handle_caregiver_followup(
                                        session,
                                        chat_id,
                                        txt,
                                        user_name,
                                        via_voice=False,
                                    )
                                else:
                                    await self.handle_text_query(session, chat_id, txt, user_name)
                    except Exception as exc:
                        logger.exception("[DEV-BOT] Update handler failed: %s", exc)
                        try:
                            # try to send a fallback error if chat_id is known
                            chat_id = int(msg["chat"]["id"])
                            await self.send_text(
                                session, chat_id,
                                "❌ <b>দুঃখিত, একটি অভ্যন্তরীণ ত্রুটি ঘটেছে।</b>\n"
                                "অনুগ্রহ করে কিছুক্ষণ পর আবার চেষ্টা করুন, অথবা /help লিখুন।"
                            )
                        except Exception:
                            pass

            while True:
                try:
                    data = await self.api_call(
                        session,
                        "getUpdates",
                        json={"offset": self.offset, "timeout": 25},
                    )
                    if not data.get("ok") and data.get("error_code") == 409:
                        logger.warning(
                            "[DEV-BOT] Another Telegram bot instance is already polling this token (409 Conflict). Backing off for 15s..."
                        )
                        await asyncio.sleep(15)
                        continue
                    for upd in data.get("result", []):
                        self.offset = int(upd["update_id"]) + 1
                        msg: Optional[Dict[str, Any]] = upd.get("message")
                        if not msg:
                            continue
                        task = asyncio.create_task(_dispatch_update(msg))
                        active_tasks.add(task)
                        task.add_done_callback(active_tasks.discard)
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.error("[DEV-BOT] Polling error: %s", exc)
                    await asyncio.sleep(3)


if __name__ == "__main__":
    bot = OushudhCheckTelegramBot()
    asyncio.run(bot.poll_forever())
