"""
OushudhCheck Telegram Bot (@OushudhCheck_bot)
Multi-Strip Medicine Safety Scanner & Bangla Voice Medication Assistant.

Features:
- Loads TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_USERNAME from .env
- Live 5-Stage Progress Feedback (`editMessageText` + `sendChatAction`) so the user sees every pipeline step in real time
- 100% Bangla Beginner Summary (`সহজ বাংলা সারসংক্ষেপ`), Morning/Noon/Night schedule, and duplicate/interaction warnings
- Supports photo scans (1–6 strips), Bangla voice questions (`faster-whisper` ASR), and text queries
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

from backend.app.services.ocr import MedicineStripScanner, evaluate_multi_strip_safety
from backend.app.services.voice import synthesize_bangla_speech, transcribe_bangla_audio

MAX_TELEGRAM_MSG_LEN = 3900
MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024  # 15 MB


def _esc(val: Any) -> str:
    return html.escape(str(val), quote=False)


def _format_tg_summary(summary: str, indent: str = "") -> str:
    """Escapes summary for Telegram HTML and always places 'সতর্কতা:' on a clean new line."""
    escaped = _esc(summary.strip())
    return re.sub(
        r"[ \t]*\n?[ \t]*(সতর্কতা\s*:)",
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
                chunks.append(block[: max_len - 3] + "...")
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

    async def api_call(self, session: aiohttp.ClientSession, method: str, **kwargs: Any) -> Dict[str, Any]:
        url = f"{API_BASE}/{method}"
        async with session.post(url, **kwargs) as resp:
            data = await resp.json()
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
        form.add_field(
            "audio",
            audio_path.read_bytes(),
            filename=audio_path.name,
            content_type="audio/mpeg",
        )
        await self.api_call(session, "sendAudio", data=form)

    async def register_bot_commands(self, session: aiohttp.ClientSession) -> None:
        commands = [
            {"command": "start", "description": "স্বাগতম ও মূল মেনু (Welcome & Menu)"},
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
            "২. 🎤 <b>বাংলা ভয়েস মেসেজ:</b> যেকোনো ওষুধ সম্পর্কে বাংলায় ভয়েস মেসেজ পাঠিয়ে প্রশ্ন করুন।\n"
            "৩. ⌨️ অথবা যেকোনো ওষুধের নাম লিখে পাঠান (যেমন: <code>Napa Extra</code>, <code>Seclo 20</code>, <code>Amdocal 5</code>, <code>Tufnil 200</code>)।\n\n"
            "📌 <b>মেনু কমান্ডসমূহ:</b>\n"
            "• /help — বিস্তারিত ব্যবহারের নিয়মাবলী\n"
            "• /guide — স্পষ্ট ছবি তোলার টিপস\n"
            "• /features — AI পাইপলাইন ও সুবিধাসমূহ\n"
            "• /safety — জরুরি চিকিৎসা সতর্কতা ও হটলাইন\n"
            "• /about — প্রজেক্ট ও টিম পরিচিতি"
        )
        await self.send_text(session, chat_id, welcome)

    async def handle_help(self, session: aiohttp.ClientSession, chat_id: int) -> None:
        msg = (
            "📘 <b>OushudhCheck ব্যবহারের নিয়মাবলী (/help)</b>\n\n"
            "<b>১. ওষুধের পাতার ছবি স্ক্যান (১–৬টি স্ট্রিপ):</b>\n"
            "• একসাথে ১ থেকে ৬টি ওষুধের পাতা সমতল টেবিলে রেখে ছবি পাঠান।\n"
            "• আমি প্রতিটি ওষুধের নাম, মাত্রা (mg), জেনেরিক, খাওয়ার নিয়ম, সহজ বাংলা সারসংক্ষেপ এবং ডুপ্লিকেট/আন্তঃক্রিয়া সতর্কতা বাংলা ভয়েসসহ জানাবো।\n\n"
            "<b>২. টেক্সট লিখে খুঁজুন:</b>\n"
            "• সরাসরি ওষুধের ব্র্যান্ড বা জেনেরিক নাম লিখুন — যেমন: <code>Napa Extra</code>, <code>Seclo</code>, <code>Omeprazole</code>, <code>Amdocal 5</code>।\n\n"
            "<b>৩. বাংলা ভয়েস মেসেজ:</b>\n"
            "• মাইক্রোফোন চেপে ওষুধের নাম বলুন (যেমন: “নাপা এক্সট্রা” বা “সেকলো ২০”) — <b>faster-whisper</b> শুনে উত্তর দেবে।\n\n"
            "👉 স্পষ্ট ছবি তোলার টিপস দেখতে /guide চাপুন।"
        )
        await self.send_text(session, chat_id, msg)

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
            "• 🔍 <b>মাল্টি-স্ট্রিপ ডিটেকশন (YOLO26n ONNX):</b> এক ছবিতেই ১–৬টি ওষুধের পাতা শনাক্ত করে।\n"
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
            "🏥 <b>OushudhCheck — Know Your Medicines. Protect Your Health.</b>\n\n"
            "✨ <b>AI-Powered Medicine Safety &amp; Bangla Voice Assistant</b>\n\n"
            "👨‍💻 <b>Team OushudhCheck:</b>\n"
            "• <b>Jubayer Ahmed</b> — Frontend &amp; AI Product Engineer\n"
            "• <b>Ridoy Hossan Rony</b> — Backend &amp; ML Systems Engineer\n\n"
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
            "⏳ <b>[ধাপ ১/৫]</b> ছবি ডাউনলোড ও <b>YOLO26n ONNX</b> স্ট্রিপ ডিটেকশন চলছে...\n"
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
        await asyncio.to_thread(self.scanner.detector.detect_strips, image)

        # Stage 2/5 Progress
        await self.edit_text(
            session,
            chat_id,
            progress_msg_id,
            "⏳ <b>[ধাপ ২/৫]</b> ফয়েল গ্লেয়ার রিমুভাল (<b>CLAHE</b>) ও <b>PP-OCRv6</b> টেক্সট এক্সট্রাকশন চলছে...\n"
            "<code>[■■■■□□□□□□] 45%</code>",
        )
        ocr_lines = await asyncio.to_thread(self.scanner.ocr_engine.extract_lines, image, True)
        if not ocr_lines:
            enhanced = await asyncio.to_thread(self.scanner.detector.preprocess_for_ocr, image)
            ocr_lines = await asyncio.to_thread(self.scanner.ocr_engine.extract_lines, enhanced, True)

        # Stage 3/5 Progress
        await self.edit_text(
            session,
            chat_id,
            progress_msg_id,
            f"⏳ <b>[ধাপ ৩/৫]</b> {len(ocr_lines)}টি টেক্সট লাইন পাওয়া গেছে। <b>RapidFuzz + ১৭,১৯১টি ওষুধের ডাটাবেস</b> ম্যাচিং চলছে...\n"
            "<code>[■■■■■■■□□□] 70%</code>",
        )

        def _cluster_and_match():
            clusters = self.scanner._cluster_ocr_lines_into_strips(ocr_lines, img_w=w, img_h=h)
            results = []
            for idx, (b_low, sc, meth, lns, bbox) in enumerate(clusters, start=1):
                raw_text = "\n".join(line.text for line in lns)
                res = self.scanner.matcher.match_strip_text(
                    raw_text=raw_text,
                    ocr_lines=lns,
                    strip_index=idx,
                    bbox=bbox,
                    forced_brand_lower=b_low if b_low else None,
                    forced_score=sc,
                    forced_method=meth,
                    slm_engine=self.scanner.slm,
                )
                results.append(res)
            return results

        strip_results = await asyncio.to_thread(_cluster_and_match)

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
            f"⏳ <b>[ধাপ ৪/৫]</b> {len(matched)}টি ওষুধ শনাক্ত হয়েছে! <b>Gemma 3 1B Bangla SLM</b> বাংলা সারসংক্ষেপ ও ডুপ্লিকেট/আন্তঃক্রিয়া যাচাই চলছে...\n"
            "<code>[■■■■■■■■■□] 90%</code>",
        )
        alerts, _, _ = evaluate_multi_strip_safety(strip_results)

        dup_ids = {sid for a in alerts if a.alert_type == "duplicate_generic" for sid in a.involved_strips}
        inter_ids = {sid for a in alerts if a.alert_type != "duplicate_generic" for sid in a.involved_strips}

        def _build_all_guidance():
            out = []
            for s in matched:
                out.append(
                    self.scanner.slm.build_schedule_and_guidance(
                        s,
                        is_duplicate=(s.strip_index in dup_ids),
                        is_interaction=(s.strip_index in inter_ids),
                    )
                )
            return out

        guidance_list = await asyncio.to_thread(_build_all_guidance)
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
                f"<b>{idx}. {_esc(m.brand_name)}</b> — <code>{_esc(strength)}</code> ({_esc(form)})\n"
                f"   • <b>জেনেরিক:</b> {_esc(m.generic)}\n"
                f"   • <b>প্রস্তুতকারক:</b> {_esc(m.manufacturer or 'N/A')}\n"
                f"   • <b>খাওয়ার সময়সূচী:</b> <code>{_esc(timing_tag)}</code>\n"
                f"   • <b>নির্দেশনা:</b> {_esc(dosage_bn)}\n"
                f"   • <b>বাংলা সারসংক্ষেপ:</b> {_format_tg_summary(bangla_summary, indent='   ')}\n"
            )
            voice_parts.append(f"স্ট্রিপ {idx}: {m.brand_name}, মাত্রা {strength}। {dosage_bn}")

        if alerts:
            lines_out.append("\n🚨 <b>সতর্কতা (Safety Alerts):</b>")
            for a in alerts:
                lines_out.append(f"• ⚠️ <b>{_esc(a.message_bn)}</b>")
                voice_parts.append(a.message_bn)
        else:
            lines_out.append("✅ <b>নিরাপত্তা যাচাই:</b> শনাক্তকৃত ওষুধে কোনো ডুপ্লিকেট জেনেরিক বা ঝুঁকিপূর্ণ আন্তঃক্রিয়া পাওয়া যায়নি।")
            voice_parts.append("ওষুধগুলোতে কোনো ডুপ্লিকেট উপাদান পাওয়া যায়নি।")

        # Remove the temporary progress bar cleanly before sending the final report so it never stays stuck at 70%/90%!
        await self.delete_message(session, chat_id, progress_msg_id)
        report_chunks = _chunk_html_blocks(lines_out)
        for chunk in report_chunks:
            await self.send_text(session, chat_id, chunk)

        # Stage 5/5 Voice Generation: Show a separate live "Voice is generating..." status message until audio arrives!
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

    async def handle_voice_message(
        self,
        session: aiohttp.ClientSession,
        chat_id: int,
        file_id: str,
        user_name: str,
    ) -> None:
        logger.info("[DEV-BOT] Received voice message from %s (chat_id=%d, file_id=%s)", user_name, chat_id, file_id)
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
        tmp_ogg.write_bytes(voice_bytes)

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

        await self.edit_text(
            session,
            chat_id,
            p_id,
            f"🗣️ <b>আপনি বলেছেন:</b> <i>“{_esc(transcribed)}”</i>\n🔍 ডাটাবেস থেকে তথ্য খোঁজা হচ্ছে...",
        )
        await self.handle_text_query(session, chat_id, transcribed, user_name, existing_msg_id=p_id)

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
                f"🔍 <code>{_esc(text)}</code> ওষুধের তথ্য খোঁজা হচ্ছে (<b>RapidFuzz + Safety SLM</b>)...",
            )

        res = await asyncio.to_thread(
            self.scanner.matcher.match_strip_text,
            text,
            None,
            1,
            None,
            None,
            100.0,
            "exact_brand",
            self.scanner.slm,
        )
        m = res.matched_medicine
        if not m:
            await self.edit_text(
                session,
                chat_id,
                existing_msg_id,
                f"🔍 <code>{_esc(text)}</code> নামে কোনো ওষুধ ডাটাবেসে পাওয়া যায়নি। সঠিক বানান লিখুন অথবা ওষুধের পাতার ছবি পাঠান।",
            )
            return

        strength = res.extracted_strength or m.package_container or "Standard"
        dosage_bn, bangla_summary, timing_tag = await asyncio.to_thread(
            self.scanner.slm.build_schedule_and_guidance,
            res,
            False,
            False,
        )
        reply = (
            f"💊 <b>{_esc(m.brand_name)}</b> — <code>{_esc(strength)}</code> ({_esc(m.dosage_form or 'Tablet')})\n"
            f"• <b>জেনেরিক:</b> {_esc(m.generic)}\n"
            f"• <b>শ্রেণী:</b> {_esc(m.drug_class or 'N/A')}\n"
            f"• <b>প্রস্তুতকারক:</b> {_esc(m.manufacturer or 'N/A')}\n"
            f"• <b>খাওয়ার সময়সূচী:</b> <code>{_esc(timing_tag)}</code>\n"
            f"• <b>নির্দেশনা:</b> {_esc(dosage_bn)}\n\n"
            f"📖 <b>সহজ বাংলা সারসংক্ষেপ:</b>\n{_format_tg_summary(bangla_summary)}"
        )
        await self.delete_message(session, chat_id, existing_msg_id)
        await self.send_text(session, chat_id, reply)

        voice_status_id = await self.send_text(
            session,
            chat_id,
            "🔊 <i>বাংলা ভয়েস অডিও তৈরি হচ্ছে... (Generating Bangla Voice...)</i>",
        )
        await self.send_chat_action(session, chat_id, "record_voice")
        audio_path = await synthesize_bangla_speech(f"{dosage_bn} {bangla_summary}")
        if audio_path and audio_path.exists():
            await self.send_voice_mp3(session, chat_id, audio_path, caption=f"🔊 {m.brand_name} ({strength})")
            await self.delete_message(session, chat_id, voice_status_id)
        else:
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
                            if cmd == "/start":
                                await self.handle_start(session, chat_id, user_name)
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
                                await self.handle_text_query(session, chat_id, txt, user_name)
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.error("[DEV-BOT] Polling error: %s", exc)
                    await asyncio.sleep(3)


if __name__ == "__main__":
    bot = OushudhCheckTelegramBot()
    asyncio.run(bot.poll_forever())
