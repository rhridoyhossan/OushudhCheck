"""
OushudhCheck Telegram Bot (@OushudhCheck_bot)
Multi-Strip Medicine Safety Scanner & Bangla Voice Medication Assistant.

Features:
- Loads TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_USERNAME from .env
- Accepts single or multi-strip medicine photos (1 to 6 strips)
- Runs RapidOCR (PP-OCRv6 ONNX) + strength extraction ('500 mg', '50 mg', '10 mg', etc.) + RapidFuzz against medicines.db
- Sends structured Bangla & English safety report + Beginner Summary
- Synthesizes and sends Bangla Voice Audio (.mp3) via Microsoft Edge TTS (`bn-BD-NabanitaNeural`)
- Includes full [DEV-BOT] logging for development & debugging
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import asyncio
import logging
import os
import sys
import time

import aiohttp
import cv2
import numpy as np
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

from backend.app.services.ocr import MedicineStripScanner
from backend.app.services.voice import synthesize_bangla_speech

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

    async def send_text(self, session: aiohttp.ClientSession, chat_id: int, text: str) -> None:
        await self.api_call(
            session,
            "sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
            },
        )

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

    async def handle_start(self, session: aiohttp.ClientSession, chat_id: int, user_name: str) -> None:
        welcome = (
            f"👋 <b>স্বাগতম {user_name}!</b> আমি <b>OushudhCheck AI</b> (@{TELEGRAM_BOT_USERNAME})।\n\n"
            "📸 <b>কীভাবে ব্যবহার করবেন:</b>\n"
            "১. টেবিলের ওপর ১ থেকে ৬টি ওষুধের পাতা (Medicine Strip) রেখে একটি স্পষ্ট ছবি তুলে পাঠান।\n"
            "২. অথবা যেকোনো ওষুধের নাম লিখে পাঠান (যেমন: <code>Napa 500mg</code>, <code>Sergel 20</code>, <code>Orthogen</code>)।\n\n"
            "✅ আমি সাথে সাথে ওষুধের <b>নাম</b>, <b>মাত্রা (mg)</b>, <b>জেনেরিক উপাদান</b>, <b>সহজ সারসংক্ষেপ</b> এবং "
            "<b>ডুপ্লিকেট বা আন্তঃক্রিয়া সতর্কতা</b> বাংলা ভয়েসসহ জানিয়ে দেবো!"
        )
        await self.send_text(session, chat_id, welcome)

    async def handle_photo(
        self,
        session: aiohttp.ClientSession,
        chat_id: int,
        file_id: str,
        user_name: str,
    ) -> None:
        t0 = time.perf_counter()
        logger.info("[DEV-BOT] Received photo from %s (chat_id=%d, file_id=%s)", user_name, chat_id, file_id)
        await self.send_text(session, chat_id, "🔍 <i>ওষুধের পাতা স্ক্যান করা হচ্ছে (RapidOCR + RapidFuzz)...</i>")

        file_info = await self.api_call(session, "getFile", json={"file_id": file_id})
        file_path = file_info.get("result", {}).get("file_path")
        if not file_path:
            await self.send_text(session, chat_id, "❌ ছবিটি ডাউনলোড করা যায়নি। অনুগ্রহ করে আবার চেষ্টা করুন।")
            return

        async with session.get(f"{FILE_BASE}/{file_path}") as dl_resp:
            img_bytes = await dl_resp.read()

        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            await self.send_text(session, chat_id, "❌ ছবির ফরম্যাট পড়া যায়নি। স্পষ্ট JPEG/PNG ছবি পাঠান।")
            return

        scan_resp = self.scanner.scan_image(image)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        matched = [s for s in scan_resp.strips if s.matched_medicine is not None]
        if not matched:
            await self.send_text(
                session,
                chat_id,
                "⚠️ ছবি থেকে কোনো ওষুধের নাম স্পষ্টভাবে পড়া যায়নি। অনুগ্রহ করে পর্যাপ্ত আলোতে ওষুধের পাতার লেখার দিকটি সামনে রেখে ছবি তুলুন।",
            )
            return

        lines_out = [
            f"💊 <b>OushudhCheck স্ক্যান রিপোর্ট</b> (<i>{len(matched)}টি স্ট্রিপ • {elapsed_ms}ms</i>)\n"
        ]
        voice_parts = [f"আপনার ছবিতে {len(matched)}টি ওষুধের পাতা শনাক্ত করা হয়েছে।"]

        for idx, s in enumerate(matched, start=1):
            m = s.matched_medicine
            assert m is not None
            strength = s.extracted_strength or m.package_container or "N/A"
            form = s.detected_dosage_form or m.dosage_form or "Tablet"
            summary = m.beginner_summary or m.indication or "চিকিৎসকের পরামর্শ অনুযায়ী সেব্য।"
            lines_out.append(
                f"<b>{idx}. {m.brand_name}</b> — <code>{strength}</code> ({form})\n"
                f"   • <b>জেনেরিক:</b> {m.generic}\n"
                f"   • <b>প্রস্তুতকারক:</b> {m.manufacturer or 'N/A'}\n"
                f"   • <b>কাজ ও নির্দেশনা:</b> {summary[:220]}\n"
            )
            voice_parts.append(
                f"স্ট্রিপ {idx}: {m.brand_name}, মাত্রা {strength}। এর জেনেরিক উপাদান {m.generic}।"
            )

        if scan_resp.safety_alerts:
            lines_out.append("🚨 <b>সতর্কতা (Safety Alerts):</b>")
            for a in scan_resp.safety_alerts:
                lines_out.append(f"• ⚠️ <b>{a.message_bn}</b>\n  <i>({a.message_en})</i>")
                voice_parts.append(a.message_bn)
        else:
            lines_out.append("✅ <b>নিরাপত্তা যাচাই:</b> শনাক্তকৃত ওষুধে কোনো ডুপ্লিকেট জেনেরিক বা ঝুঁকিপূর্ণ আন্তঃক্রিয়া পাওয়া যায়নি।")
            voice_parts.append("শনাক্তকৃত ওষুধে কোনো ডুপ্লিকেট উপাদান পাওয়া যায়নি।")

        report_html = "\n".join(lines_out)
        await self.send_text(session, chat_id, report_html)

        voice_script = " ".join(voice_parts)
        audio_path = await synthesize_bangla_speech(voice_script)
        if audio_path and audio_path.exists():
            await self.send_voice_mp3(
                session,
                chat_id,
                audio_path,
                caption="🔊 বাংলা ভয়েস গাইড (OushudhCheck AI)",
            )

    async def handle_text_query(self, session: aiohttp.ClientSession, chat_id: int, text: str, user_name: str) -> None:
        logger.info("[DEV-BOT] Received text query from %s: %r", user_name, text)
        res = self.scanner.matcher.match_strip_text(raw_text=text, strip_index=1)
        m = res.matched_medicine
        if not m:
            await self.send_text(
                session,
                chat_id,
                f"🔍 <code>{text}</code> নামে কোনো ওষুধ ডাটাবেসে পাওয়া যায়নি। সঠিক বানান লিখুন অথবা ওষুধের পাতার ছবি পাঠান।",
            )
            return

        strength = res.extracted_strength or m.package_container or "Standard"
        summary = m.beginner_summary or m.indication_description or m.indication or "N/A"
        reply = (
            f"💊 <b>{m.brand_name}</b> — <code>{strength}</code> ({m.dosage_form or 'Tablet'})\n"
            f"• <b>জেনেরিক:</b> {m.generic}\n"
            f"• <b>শ্রেণী:</b> {m.drug_class or 'N/A'}\n"
            f"• <b>প্রস্তুতকারক:</b> {m.manufacturer or 'N/A'}\n\n"
            f"📖 <b>সহজ সারসংক্ষেপ:</b>\n{summary[:500]}"
        )
        await self.send_text(session, chat_id, reply)

        voice_text = (
            f"{m.brand_name}, মাত্রা {strength}। এর জেনেরিক নাম {m.generic}, প্রস্তুতকারক {m.manufacturer or ''}। "
            "চিকিৎসকের পরামর্শ অনুযায়ী ওষুধটি সেবন করুন।"
        )
        audio_path = await synthesize_bangla_speech(voice_text)
        if audio_path and audio_path.exists():
            await self.send_voice_mp3(session, chat_id, audio_path, caption=f"🔊 {m.brand_name} ({strength})")

    async def poll_forever(self) -> None:
        async with aiohttp.ClientSession() as session:
            me = await self.api_call(session, "getMe")
            if me.get("ok"):
                bot_info = me["result"]
                logger.info(
                    "[DEV-BOT] Connected to Telegram Bot @%s (id=%s, name=%r)",
                    bot_info.get("username"),
                    bot_info.get("id"),
                    bot_info.get("first_name"),
                )
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
                        elif "text" in msg:
                            txt = str(msg["text"]).strip()
                            if txt.startswith("/start") or txt.startswith("/help"):
                                await self.handle_start(session, chat_id, user_name)
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
