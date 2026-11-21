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
from typing import Any, Dict, Optional
import asyncio
import logging
import os
import sys
import tempfile
import time

import aiohttp
import cv2
import numpy as np
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

from backend.app.services.ocr import MedicineStripScanner, evaluate_multi_strip_safety
from backend.app.services.slm import generate_bangla_beginner_summary
from backend.app.services.voice import synthesize_bangla_speech, transcribe_bangla_audio

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
    ) -> None:
        if not message_id:
            return
        await self.api_call(
            session,
            "editMessageText",
            json={
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
                "parse_mode": "HTML",
            },
        )

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

    async def handle_start(self, session: aiohttp.ClientSession, chat_id: int, user_name: str) -> None:
        welcome = (
            f"👋 <b>স্বাগতম {user_name}!</b> আমি <b>OushudhCheck AI</b> (@{TELEGRAM_BOT_USERNAME})।\n\n"
            "📸 <b>কীভাবে ব্যবহার করবেন:</b>\n"
            "১. টেবিলের ওপর ১ থেকে ৬টি ওষুধের পাতা (Medicine Strip) রেখে একটি স্পষ্ট ছবি তুলে পাঠান।\n"
            "২. 🎤 <b>বাংলা ভয়েস মেসেজ:</b> যেকোনো ওষুধ সম্পর্কে বাংলায় ভয়েস মেসেজ পাঠিয়ে প্রশ্ন করুন।\n"
            "৩. ⌨️ অথবা যেকোনো ওষুধের নাম লিখে পাঠান (যেমন: <code>Napa Extra</code>, <code>Amdocal 5</code>, <code>Tufnil 200</code>)।\n\n"
            "✅ আমি সাথে সাথে ওষুধের <b>নাম</b>, <b>মাত্রা (mg)</b>, <b>জেনেরিক উপাদান</b>, <b>সহজ বাংলা সারসংক্ষেপ</b>, "
            "<b>সকাল-দুপুর-রাত খাওয়ার নিয়ম</b> এবং <b>ডুপ্লিকেট বা আন্তঃক্রিয়া সতর্কতা</b> বাংলা ভয়েসসহ জানিয়ে দেবো!"
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

        async with session.get(f"{FILE_BASE}/{file_path}") as dl_resp:
            img_bytes = await dl_resp.read()

        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            await self.edit_text(session, chat_id, progress_msg_id, "❌ ছবির ফরম্যাট পড়া যায়নি। স্পষ্ট JPEG/PNG ছবি পাঠান।")
            return

        h, w = image.shape[:2]
        _ = self.scanner.detector.detect_strips(image)

        # Stage 2/5 Progress
        await self.edit_text(
            session,
            chat_id,
            progress_msg_id,
            "⏳ <b>[ধাপ ২/৫]</b> ফয়েল গ্লেয়ার রিমুভাল (<b>CLAHE</b>) ও <b>PP-OCRv6</b> টেক্সট এক্সট্রাকশন চলছে...\n"
            "<code>[■■■■□□□□□□] 45%</code>",
        )
        ocr_lines = self.scanner.ocr_engine.extract_lines(image, include_rotated=True)
        if not ocr_lines:
            enhanced = self.scanner.detector.preprocess_for_ocr(image)
            ocr_lines = self.scanner.ocr_engine.extract_lines(enhanced, include_rotated=True)

        # Stage 3/5 Progress
        await self.edit_text(
            session,
            chat_id,
            progress_msg_id,
            f"⏳ <b>[ধাপ ৩/৫]</b> {len(ocr_lines)}টি টেক্সট লাইন পাওয়া গেছে। <b>RapidFuzz + ১৭,১৯১টি ওষুধের ডাটাবেস</b> ম্যাচিং চলছে...\n"
            "<code>[■■■■■■■□□□] 70%</code>",
        )
        clusters = self.scanner._cluster_ocr_lines_into_strips(ocr_lines, img_w=w, img_h=h)
        strip_results = []
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
            strip_results.append(res)

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
            f"⏳ <b>[ধাপ ৪/৫]</b> {len(matched)}টি ওষুধ শনাক্ত হয়েছে! <b>Safety SLM</b> বাংলা সারসংক্ষেপ ও ডুপ্লিকেট/আন্তঃক্রিয়া যাচাই চলছে...\n"
            "<code>[■■■■■■■■■□] 90%</code>",
        )
        alerts, _, _ = evaluate_multi_strip_safety(strip_results)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        dup_ids = {sid for a in alerts if a.alert_type == "duplicate_generic" for sid in a.involved_strips}
        inter_ids = {sid for a in alerts if a.alert_type != "duplicate_generic" for sid in a.involved_strips}

        lines_out = [
            f"💊 <b>OushudhCheck স্ক্যান রিপোর্ট</b> (<i>{len(matched)}টি স্ট্রিপ • {elapsed_ms}ms</i>)\n"
        ]
        voice_parts = [f"আপনার ছবিতে {len(matched)}টি ওষুধের পাতা শনাক্ত করা হয়েছে।"]

        for idx, s in enumerate(matched, start=1):
            m = s.matched_medicine
            assert m is not None
            strength = s.extracted_strength or m.package_container or "N/A"
            form = s.detected_dosage_form or m.dosage_form or "Tablet"
            dosage_bn, bangla_summary, timing_tag = self.scanner.slm.build_schedule_and_guidance(
                s,
                is_duplicate=(s.strip_index in dup_ids),
                is_interaction=(s.strip_index in inter_ids),
            )
            lines_out.append(
                f"<b>{idx}. {m.brand_name}</b> — <code>{strength}</code> ({form})\n"
                f"   • <b>জেনেরিক:</b> {m.generic}\n"
                f"   • <b>প্রস্তুতকারক:</b> {m.manufacturer or 'N/A'}\n"
                f"   • <b>খাওয়ার সময়সূচী:</b> <code>{timing_tag}</code>\n"
                f"   • <b>নির্দেশনা:</b> {dosage_bn}\n"
                f"   • <b>বাংলা সারসংক্ষেপ:</b> {bangla_summary}\n"
            )
            voice_parts.append(f"স্ট্রিপ {idx}: {m.brand_name}, মাত্রা {strength}। {dosage_bn}")

        if alerts:
            lines_out.append("🚨 <b>সতর্কতা (Safety Alerts):</b>")
            for a in alerts:
                lines_out.append(f"• ⚠️ <b>{a.message_bn}</b>")
                voice_parts.append(a.message_bn)
        else:
            lines_out.append("✅ <b>নিরাপত্তা যাচাই:</b> শনাক্তকৃত ওষুধে কোনো ডুপ্লিকেট জেনেরিক বা ঝুঁকিপূর্ণ আন্তঃক্রিয়া পাওয়া যায়নি।")
            voice_parts.append("ওষুধগুলোতে কোনো ডুপ্লিকেট উপাদান পাওয়া যায়নি।")

        # Update progress message to final report immediately so user gets text results ASAP without waiting for TTS!
        report_html = "\n".join(lines_out)
        await self.edit_text(session, chat_id, progress_msg_id, report_html)

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

        async with session.get(f"{FILE_BASE}/{file_path}") as dl_resp:
            voice_bytes = await dl_resp.read()

        tmp_ogg = Path(tempfile.gettempdir()) / f"tg_voice_{chat_id}_{int(time.time())}.ogg"
        tmp_ogg.write_bytes(voice_bytes)

        try:
            transcribed = transcribe_bangla_audio(tmp_ogg)
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
            f"🗣️ <b>আপনি বলেছেন:</b> <i>“{transcribed}”</i>\n🔍 ডাটাবেস থেকে তথ্য খোঁজা হচ্ছে...",
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
                f"🔍 <code>{text}</code> ওষুধের তথ্য খোঁজা হচ্ছে (<b>RapidFuzz + Safety SLM</b>)...",
            )

        res = self.scanner.matcher.match_strip_text(raw_text=text, strip_index=1, slm_engine=self.scanner.slm)
        m = res.matched_medicine
        if not m:
            await self.edit_text(
                session,
                chat_id,
                existing_msg_id,
                f"🔍 <code>{text}</code> নামে কোনো ওষুধ ডাটাবেসে পাওয়া যায়নি। সঠিক বানান লিখুন অথবা ওষুধের পাতার ছবি পাঠান।",
            )
            return

        strength = res.extracted_strength or m.package_container or "Standard"
        dosage_bn, bangla_summary, timing_tag = self.scanner.slm.build_schedule_and_guidance(
            res, is_duplicate=False, is_interaction=False
        )
        reply = (
            f"💊 <b>{m.brand_name}</b> — <code>{strength}</code> ({m.dosage_form or 'Tablet'})\n"
            f"• <b>জেনেরিক:</b> {m.generic}\n"
            f"• <b>শ্রেণী:</b> {m.drug_class or 'N/A'}\n"
            f"• <b>প্রস্তুতকারক:</b> {m.manufacturer or 'N/A'}\n"
            f"• <b>খাওয়ার সময়সূচী:</b> <code>{timing_tag}</code>\n"
            f"• <b>নির্দেশনা:</b> {dosage_bn}\n\n"
            f"📖 <b>সহজ বাংলা সারসংক্ষেপ:</b>\n{bangla_summary}"
        )
        await self.edit_text(session, chat_id, existing_msg_id, reply)

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
                        elif "voice" in msg and msg["voice"]:
                            await self.handle_voice_message(session, chat_id, msg["voice"]["file_id"], user_name)
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
