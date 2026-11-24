"""
Quantized Safety SLM & Mini AI Disambiguation Engine (Gemma 3 1B-Q4_K_M + Grounded Bangla Clinical NLG).

Responsibilities:
1. Mini-AI OCR Candidate Re-Ranking & Context Disambiguation:
   - Eliminates torn-foil substring ghost matches (e.g., 'dipine' from 'Amlodipine', 'epam' from 'Diazepam').
   - Merges split multi-word brand boxes (e.g., 'Napa®' + 'Extra' -> 'Napa Extra').
   - Resolves EDCL (Essential Drugs Co. Ltd.) government generic blister strips ('DIAZEPAM TABLETS', 'VITAMIN B-COMPLEX').
   - Validates and orders extracted dosage strengths against the active ingredient count in `medicines.db`.
2. Grounded Bangla Beginner Summary & Daily Schedule Generation (Morning / Noon / Night, Before/After Meals):
   - Generates 100% Bangla beginner summaries (`beginner_summary`), voice scripts, and daily schedule tags grounded strictly in `medicines.db`.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Sequence, Set, Tuple
import json
import logging
import os
import re
import textwrap
import threading
import time
import urllib.error
import urllib.request

from rapidfuzz import fuzz

try:
    from backend.app.core.config import PROJECT_ROOT
    from backend.app.models.schemas import MedicineInfo, SafetyAlert, StripScanResult
except ImportError:
    from app.core.config import PROJECT_ROOT  # type: ignore
    from app.models.schemas import MedicineInfo, SafetyAlert, StripScanResult  # type: ignore

logger = logging.getLogger("oushudhcheck.slm")

SLM_MODEL_NAME = os.getenv("SLM_MODEL_NAME", "oushudhcheck-gemma3-1b")
SLM_API_URL = os.getenv("SLM_API_URL", "http://127.0.0.1:8081/v1/chat/completions").strip()
SLM_TIMEOUT_SEC = float(os.getenv("SLM_TIMEOUT_SEC", "30.0"))
SLM_MAX_TOKENS = int(os.getenv("SLM_MAX_TOKENS", "768"))

_DEFAULT_SYSTEM_PROMPT = (
    "তুমি OushudhCheck মেডিকেল সহায়ক। শুধুমাত্র প্রদত্ত ডাটাবেস তথ্যের ভিত্তিতে "
    "রোগীর জন্য ৩-৪ বাক্যে সহজ ও স্পষ্ট বাংলায় ওষুধটির কাজ, খাওয়ার নিয়ম ও সময় এবং "
    "গুরুত্বপূর্ণ সতর্কতা লেখো। ওষুধের ব্র্যান্ড নাম, মাত্রা (যেমন 20 mg বা 500 mg + 65 mg) "
    "এবং জেনেরিক উপাদানের নাম ডাটাবেসে যেভাবে ইংরেজিতে আছে ঠিক সেভাবেই অপরিবর্তিত রাখবে, "
    "আর বাকি সম্পূর্ণ ব্যাখ্যা সহজ বাংলায় লিখবে। ডাটাবেসে নেই এমন কোনো মাত্রা, পার্শ্বপ্রতিক্রিয়া বা "
    "সতর্কতা মনগড়াভাবে যোগ করবে না।"
)


def _load_trained_system_prompt() -> str:
    prompt_file = PROJECT_ROOT / "models" / "system_prompt.txt"
    if prompt_file.exists():
        try:
            txt = prompt_file.read_text(encoding="utf-8").strip()
            if txt:
                return txt
        except OSError:
            pass
    return _DEFAULT_SYSTEM_PROMPT


TRAINED_SYSTEM_PROMPT = _load_trained_system_prompt()
_LAST_SLM_USED_LIVE = False
_SLM_CIRCUIT_OPEN_UNTIL = 0.0
_SLM_CIRCUIT_COOLDOWN_SEC = 30.0
_SLM_CACHE_MAX = 2048
_SLM_CACHE: "OrderedDict[Tuple[str, ...], str]" = OrderedDict()
_SLM_QA_CACHE_MAX = 512
_SLM_QA_CACHE: "OrderedDict[Tuple[str, ...], str]" = OrderedDict()
_SLM_CACHE_LOCK = threading.Lock()


_ORAL_PHRASE_RE = re.compile(
    r"(?:খালি\s*পেটে|ভরা\s*পেটে|খাবারের\s*আগে|খাবারের\s*পরে|পানি\s*দিয়ে|সেবন\s*করতে|খেতে\s*হয়|ট্যাবলেট|ক্যাপসুল)"
)


def _get_non_oral_route_rule(dosage_form: str) -> Optional[str]:
    form_low = (dosage_form or "").strip().lower()
    if "suppository" in form_low:
        return "এটি একটি সাপোজিটরি, যা পায়ুপথে ব্যবহার করতে হয়: মুখে খাওয়া যাবে না। চিকিৎসকের নির্ধারিত মাত্রা অনুযায়ী ব্যবহার করুন।"
    if any(k in form_low for k in ("cream", "ointment", "gel", "lotion", "shampoo", "spray", "paste")):
        return "এটি বাহ্যিক ব্যবহারের ওষুধ; আক্রান্ত স্থানে পরিষ্কার হাতে পাতলা করে লাগাতে হয়: মুখে খাওয়া যাবে না।"
    if any(k in form_low for k in ("drop", "ophthalmic", "otic", "nasal")):
        return "এটি ড্রপ হিসেবে ব্যবহার্য; চিকিৎসকের নির্ধারিত ফোঁটা অনুযায়ী নির্দিষ্ট স্থানে প্রয়োগ করুন: মুখে খাওয়া যাবে না।"
    if "injection" in form_low or "infusion" in form_low:
        return "এটি একটি ইনজেকশন, যা শুধুমাত্র চিকিৎসক বা প্রশিক্ষিত নার্সের তত্ত্বাবধানে প্রয়োগ করতে হবে।"
    if "inhaler" in form_low or "nebul" in form_low:
        return "এটি শ্বাসের মাধ্যমে (ইনহেলার হিসেবে) গ্রহণ করতে হয়; চিকিৎসকের দেখানো নিয়ম অনুযায়ী ব্যবহার করুন।"
    return None


def _enforce_route_safety(text: str, dosage_form: str) -> str:
    """
    Post-generation clinical safety guardrail: if the 1B SLM hallucinates oral-ingestion
    instructions ('খালি পেটে', 'ভরা পেটে', 'সেবন করতে হয়', 'ট্যাবলেট') for a non-oral
    dosage form (Suppository, Cream, Drops, Injection, Inhaler), replaces the offending
    instruction sentence(s) with the verified route-safe Bangla instruction while keeping
    the indication and warning intact.
    """
    route_rule = _get_non_oral_route_rule(dosage_form)
    if not route_rule:
        return text

    # Split main body from 'সতর্কতা:' block so warnings are preserved cleanly
    parts = re.split(r"(\n?সতর্কতা\s*:.*)", text, maxsplit=1, flags=re.DOTALL)
    body = parts[0].strip()
    warn_tail = parts[1] if len(parts) > 1 else ""

    if not _ORAL_PHRASE_RE.search(body):
        return text

    sentences = [s.strip() for s in re.split(r"(?<=।)\s*", body) if s.strip()]
    safe_sentences: List[str] = []
    inserted_route = False
    for sent in sentences:
        if _ORAL_PHRASE_RE.search(sent):
            if not inserted_route:
                safe_sentences.append(route_rule)
                inserted_route = True
        else:
            safe_sentences.append(sent)
    if not inserted_route:
        safe_sentences.append(route_rule)

    rebuilt = " ".join(safe_sentences).strip()
    if warn_tail:
        rebuilt = f"{rebuilt}\n{warn_tail.strip()}"
    return _format_bangla_summary(rebuilt)


def _format_bangla_summary(text: str) -> str:
    """Place 'সতর্কতা:' cleanly on a new line for readability across web and Telegram."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"[ \t]*\n?[ \t]*(সতর্কতা\s*:)", r"\n\1", cleaned)
    return cleaned.strip()


def get_active_slm_label() -> str:
    with _SLM_CACHE_LOCK:
        live = _LAST_SLM_USED_LIVE
    if live:
        return f"Gemma 3 1B-Q4_K_M Fine-Tuned Bangla SLM ({SLM_MODEL_NAME})"
    return "Gemma 3 1B Bangla SLM (Grounded Fallback Mode)"


def _call_local_slm_cached(
    brand_name: str,
    generic: str,
    dosage_form: str,
    manufacturer: str,
    drug_class: str,
    indication: str,
    strength: str,
    dosage_description: str,
    side_effects_description: str,
    interaction_description: str,
    precautions_description: str,
) -> Optional[str]:
    """
    Calls the local fine-tuned Gemma 3 1B GGUF server (llama-server or Ollama) via
    OpenAI-compatible `/v1/chat/completions`.
    Only caches successful Bangla responses (never caches failures/None) so if the AI server
    is temporarily offline, callers immediately fall back to the deterministic summary method
    and automatically recover once the AI server is back online.
    """
    global _SLM_CIRCUIT_OPEN_UNTIL
    if not SLM_API_URL:
        return None

    cache_key = (
        brand_name,
        generic,
        dosage_form,
        manufacturer,
        drug_class,
        indication,
        strength,
        dosage_description,
        side_effects_description,
        interaction_description,
        precautions_description,
    )
    with _SLM_CACHE_LOCK:
        cached = _SLM_CACHE.get(cache_key)
        if cached is not None:
            _SLM_CACHE.move_to_end(cache_key)
            return cached
        if time.monotonic() < _SLM_CIRCUIT_OPEN_UNTIL:
            return None

    fields = [
        ("ওষুধের নাম", brand_name, 120),
        ("মাত্রা বা প্যাক", strength, 120),
        ("ধরন", dosage_form, 80),
        ("জেনেরিক উপাদান", generic, 240),
        ("প্রস্তুতকারক", manufacturer, 120),
        ("ওষুধের শ্রেণি", drug_class, 180),
        ("ব্যবহারের নির্দেশনা", indication, 400),
        ("সেবনবিধি", dosage_description, 600),
        ("পার্শ্বপ্রতিক্রিয়া", side_effects_description, 400),
        ("অন্য ওষুধের সাথে বিক্রিয়া", interaction_description, 400),
        ("সতর্কতা ও নিষেধাজ্ঞা", precautions_description, 400),
    ]
    lines = ["[ডাটাবেস তথ্য]"]
    for label, val, limit in fields:
        clean = textwrap.shorten(str(val or "").strip(), width=limit, placeholder="...")
        if clean:
            lines.append(f"{label}: {clean}")
    non_oral_rule = _get_non_oral_route_rule(dosage_form)
    if non_oral_rule:
        lines.append(
            f"ব্যবহারের মাধ্যম (গুরুত্বপূর্ণ): {non_oral_rule} "
            "কখনোই খালি পেটে, ভরা পেটে, খাবারের আগে/পরে বা মুখে সেবন লিখবে না।"
        )
        lines.append("\nরোগীর প্রশ্ন: এই ওষুধটি কী কাজ করে, কীভাবে ব্যবহার করতে হবে এবং কী সতর্কতা মানতে হবে?")
    else:
        lines.append("\nরোগীর প্রশ্ন: এই ওষুধটি কী কাজ করে, কখন খাবো এবং কী সতর্কতা মানতে হবে?")
    user_prompt = "\n".join(lines)

    payload = {
        "model": SLM_MODEL_NAME,
        "messages": [
            {"role": "user", "content": f"{TRAINED_SYSTEM_PROMPT}\n\n{user_prompt}"},
        ],
        "temperature": 0.15,
        "top_p": 0.9,
        "repeat_penalty": 1.05,
        "max_tokens": SLM_MAX_TOKENS,
        "stream": False,
    }

    req = urllib.request.Request(
        SLM_API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=SLM_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        # Validate that the response actually contains Bangla text before accepting & caching it
        if content and re.search(r"[\u0980-\u09FF]{10,}", content):
            formatted = _enforce_route_safety(_format_bangla_summary(content), dosage_form)
            with _SLM_CACHE_LOCK:
                _SLM_CACHE[cache_key] = formatted
                _SLM_CACHE.move_to_end(cache_key)
                while len(_SLM_CACHE) > _SLM_CACHE_MAX:
                    _SLM_CACHE.popitem(last=False)
            return formatted
        return None
    except Exception as exc:
        with _SLM_CACHE_LOCK:
            _SLM_CIRCUIT_OPEN_UNTIL = time.monotonic() + _SLM_CIRCUIT_COOLDOWN_SEC
        logger.warning(
            "[DEV-SLM] Local SLM server unavailable (%s); falling back to grounded summary (cooldown %.0fs).",
            exc,
            _SLM_CIRCUIT_COOLDOWN_SEC,
        )
        return None

BRAND_MODIFIERS = {"extra", "extend", "plus", "rapid", "one", "max", "pro", "forte", "ds", "sr", "xr", "cr"}

_DOSAGE_FORM_BN: Dict[str, str] = {
    "tablet": "ট্যাবলেট (Tablet)",
    "capsule": "ক্যাপসুল (Capsule)",
    "syrup": "সিরাপ (Syrup)",
    "suspension": "সাসপেনশন (Suspension)",
    "oral suspension": "ওরাল সাসপেনশন (Oral Suspension)",
    "powder for suspension": "পাউডার ফর সাসপেনশন",
    "injection": "ইনজেকশন (Injection)",
    "im/iv injection": "ইনজেকশন (IM/IV Injection)",
    "cream": "ক্রিম (Cream)",
    "ointment": "মলম (Ointment)",
    "gel": "জেল (Gel)",
    "drop": "ড্রপ (Drop)",
    "ophthalmic solution": "চোখের ড্রপ (Eye Drops)",
    "suppository": "সাপোজিটরি (Suppository)",
    "inhaler": "ইনহেলার (Inhaler)",
}


def generate_bangla_beginner_summary(
    brand_name: str,
    generic: str,
    dosage_form: Optional[str] = None,
    manufacturer: Optional[str] = None,
    drug_class: Optional[str] = None,
    indication: Optional[str] = None,
    strength: Optional[str] = None,
    dosage_description: Optional[str] = None,
    side_effects_description: Optional[str] = None,
    interaction_description: Optional[str] = None,
    precautions_description: Optional[str] = None,
    use_llm: bool = True,
) -> str:
    """
    Generates a clear, patient-friendly Bangla summary (`সহজ বাংলা সারসংক্ষেপ`) using the
    fine-tuned Gemma 3 1B GGUF model via local `llama-server` / Ollama (`SLM_API_URL`),
    with an automatic grounded fallback if the local SLM server is offline.
    """
    global _LAST_SLM_USED_LIVE

    if use_llm and SLM_API_URL:
        llm_out = _call_local_slm_cached(
            brand_name=(brand_name or "").strip(),
            generic=(generic or "").strip(),
            dosage_form=(dosage_form or "Tablet").strip(),
            manufacturer=(manufacturer or "").strip(),
            drug_class=(drug_class or "").strip(),
            indication=(indication or "").strip(),
            strength=(strength or "").strip(),
            dosage_description=(dosage_description or "").strip(),
            side_effects_description=(side_effects_description or "").strip(),
            interaction_description=(interaction_description or "").strip(),
            precautions_description=(precautions_description or "").strip(),
        )
        if llm_out:
            with _SLM_CACHE_LOCK:
                _LAST_SLM_USED_LIVE = True
            return llm_out
        with _SLM_CACHE_LOCK:
            _LAST_SLM_USED_LIVE = False

    form_raw = (dosage_form or "Tablet").strip()
    form_low = form_raw.lower()
    form_bn = _DOSAGE_FORM_BN.get(form_low, f"{form_raw}")
    mfg_str = f" ({manufacturer})" if manufacturer else ""
    str_part = f" {strength}" if strength else ""
    gen_low = (generic or "").lower()
    ind_low = (indication or "").lower()
    cls_low = (drug_class or "").lower()

    is_suppository = "suppository" in form_low
    is_topical = any(k in form_low for k in ("cream", "ointment", "gel", "lotion", "shampoo", "spray", "paste"))
    is_drop = any(k in form_low for k in ("drop", "ophthalmic", "otic", "nasal"))
    is_injection = "injection" in form_low or "infusion" in form_low
    is_inhaler = "inhaler" in form_low or "nebul" in form_low
    is_liquid_oral = any(k in form_low for k in ("syrup", "suspension", "elixir", "solution")) and not is_drop

    # Determine primary clinical use & meal instruction in natural everyday Bangla
    if "paracetamol" in gen_low and "caffeine" in gen_low:
        use_bn = "এটি জ্বর, তীব্র মাথাব্যথা, মাইগ্রেন, দাঁতব্যথা, সর্দি-জ্বর এবং শরীরের বিভিন্ন ব্যথা দ্রুত কমাতে ব্যবহৃত হয়।"
        rule_bn = "প্রাপ্তবয়স্কদের জন্য সাধারণত ১টি ট্যাবলেট দিনে ৩ বার ভরা পেটে (খাবারের পরে) সেবন করা হয়। দিনে ৪ গ্রামের বেশি প্যারাসিটামল সেবন করা যাবে না।"
        warn_bn = "সতর্কতা: এর সাথে অন্য কোনো প্যারাসিটামল (যেমন Napa বা Ace) একসাথে খাবেন না এবং অতিরিক্ত চা-কফি পান এড়িয়ে চলুন।"
    elif "paracetamol" in gen_low:
        use_bn = "এটি জ্বর, সর্দি-জ্বর, মাথাব্যথা, দাঁতব্যথা, কানব্যথা এবং শরীরের হালকা থেকে মাঝারি ব্যথা উপশমে ব্যবহৃত হয়।"
        rule_bn = "প্রাপ্তবয়স্কদের জন্য ১টি ট্যাবলেট দিনে ৩ থেকে ৪ বার খাবারের পরে ভরা পেটে সেব্য (অথবা জ্বর/ব্যথা থাকলে ৪ থেকে ৬ ঘণ্টা পরপর)।"
        warn_bn = "সতর্কতা: একই সাথে একাধিক প্যারাসিটামল জাতীয় ওষুধ খেলে লিভারের মারাত্মক ক্ষতি (ওভারডোজ) হতে পারে।"
    elif any(k in gen_low for k in ("omeprazole", "esomeprazole", "pantoprazole", "rabeprazole", "lansoprazole", "dexlansoprazole")):
        use_bn = "এটি গ্যাস্ট্রিক, পেটে অ্যাসিডিটি, বুকজ্বালা, পেপটিক আলসার এবং ব্যথানাশক ওষুধজনিত আলসার প্রতিরোধে ব্যবহৃত হয়।"
        rule_bn = "এই ওষুধটি দিনে ১ থেকে ২ বার, অবশ্যই খাবারের ৩০ মিনিট আগে পানি দিয়ে গিলে খেতে হয়।"
        warn_bn = "সতর্কতা: ক্যাপসুল বা ট্যাবলেটটি চিবিয়ে বা ভেঙে খাবেন না। দীর্ঘদিন সেবনের ক্ষেত্রে চিকিৎসকের পরামর্শ নিন।"
    elif any(k in gen_low for k in ("naproxen", "tolfenamic", "ketorolac", "ibuprofen", "diclofenac", "aceclofenac", "etoricoxib", "meloxicam")):
        use_bn = "এটি একটি ব্যথানাশক ও প্রদাহরোধী (NSAID) ওষুধ, যা মাইগ্রেন, মাথাব্যথা, বাতের ব্যথা, মাংসপেশির ব্যথা, দাঁতব্যথা এবং মাসিকের তীব্র ব্যথা কমাতে ব্যবহৃত হয়।"
        rule_bn = "গ্যাস্ট্রিক বা পেটের জ্বালাপোড়া এড়াতে এই ওষুধটি অবশ্যই ভরা পেটে (খাবারের পরে) সেবন করতে হবে।"
        warn_bn = "সতর্কতা: খালি পেটে খাবেন না। পেপটিক আলসার, কিডনি রোগ বা হাঁপানি থাকলে চিকিৎসকের পরামর্শ ছাড়া সেবন নিষিদ্ধ।"
    elif any(k in gen_low for k in ("amlodipine", "losartan", "telmisartan", "bisoprolol", "atenolol", "olmesartan", "valsartan")):
        use_bn = "এটি উচ্চ রক্তচাপ (High Blood Pressure) নিয়ন্ত্রণে এবং বুকে ব্যথা (Angina) ও হৃদরোগের ঝুঁকি কমাতে ব্যবহৃত হয়।"
        rule_bn = "প্রতিদিন নির্দিষ্ট সময়ে (সাধারণত সকালে) ১টি করে ট্যাবলেট খাবারের আগে বা পরে নিয়মিত সেবন করতে হয়।"
        warn_bn = "সতর্কতা: রক্তচাপ স্বাভাবিক মনে হলেও চিকিৎসকের পরামর্শ ছাড়া হঠাৎ ওষুধ খাওয়া বন্ধ করবেন না।"
    elif any(k in gen_low for k in ("clonazepam", "diazepam", "alprazolam", "bromazepam", "midazolam", "nitrazepam")):
        use_bn = "এটি অতিরিক্ত মানসিক উদ্বেগ (Anxiety), প্যানিক অ্যাটাক, অনিদ্রা, খিঁচুনি (Epilepsy) এবং মাংসপেশির খিঁচুনি নিয়ন্ত্রণে ব্যবহৃত হয়।"
        rule_bn = "এটি সাধারণত রাতে ঘুমানোর আগে অথবা চিকিৎসকের নির্ধারিত মাত্রায় সেবন করা হয়।"
        warn_bn = "সতর্কতা: এই ওষুধ খেলে ঘুম বা তন্দ্রাভাব হতে পারে, তাই সেবনের পর গাড়ি চালানো বা ঝুঁকিপূর্ণ কাজ থেকে বিরত থাকুন। চিকিৎসকের প্রেসক্রিপশন ছাড়া দীর্ঘদিন খাবেন না।"
    elif "flunarizine" in gen_low:
        use_bn = "এটি মাইগ্রেন প্রতিরোধে এবং মাথা ঘোরা (Vertigo) ও ভেস্টিবুলার সমস্যার চিকিৎসায় ব্যবহৃত হয়।"
        rule_bn = "সাধারণত রাতে ঘুমানোর আগে প্রতিদিন ১টি করে ট্যাবলেট সেবন করার পরামর্শ দেওয়া হয়।"
        warn_bn = "সতর্কতা: এতে হালকা ঘুম ঘুম ভাব বা ওজন বৃদ্ধি হতে পারে; চিকিৎসকের পরামর্শ অনুযায়ী নির্দিষ্ট মেয়াদে সেবন করুন।"
    elif "sertraline" in gen_low or "escitalopram" in gen_low or "fluoxetine" in gen_low or "amitriptyline" in gen_low:
        use_bn = "এটি মানসিক অবসাদ (Depression), অতিরিক্ত দুশ্চিন্তা, ওসিডি (OCD) এবং প্যানিক ডিসঅর্ডারের চিকিৎসায় ব্যবহৃত হয়।"
        rule_bn = "প্রতিদিন সকালে অথবা রাতে নির্দিষ্ট সময়ে খাবারের সাথে বা খাবারের পরে সেবন করতে হয়।"
        warn_bn = "সতর্কতা: চিকিৎসকের পরামর্শ ছাড়া হঠাৎ এই ওষুধের মাত্রা কমানো বা বন্ধ করা যাবে না।"
    elif "pregabalin" in gen_low or "gabapentin" in gen_low:
        use_bn = "এটি ডায়াবেটিস বা আঘাতজনিত স্নায়ুর ব্যথা (Neuropathic Pain), ফাইব্রোমায়ালজিয়া এবং খিঁচুনি নিয়ন্ত্রণে ব্যবহৃত হয়।"
        rule_bn = "চিকিৎসকের পরামর্শ অনুযায়ী দিনে ১-২ বার খাবারের পরে সেবন করা হয়।"
        warn_bn = "সতর্কতা: ওষুধটি সেবনে মাথা ঘোরা বা ঘুম ভাব হতে পারে; চিকিৎসকের পরামর্শ ছাড়া হঠাৎ বন্ধ করবেন না।"
    elif "ondansetron" in gen_low or "domperidone" in gen_low or "palonosetron" in gen_low:
        use_bn = "এটি বমি এবং বমি বমি ভাব (Nausea & Vomiting) দ্রুত বন্ধ করতে বা প্রতিরোধ করতে ব্যবহৃত হয়।"
        rule_bn = "সাধারণত খাবারের ১৫-৩০ মিনিট আগে অথবা বমি ভাব হলে চিকিৎসকের পরামর্শ অনুযায়ী সেব্য।"
        warn_bn = "সতর্কতা: কোষ্ঠকাঠিন্য বা মাথাব্যথা দেখা দিলে এবং লক্ষণ না কমলে চিকিৎসকের পরামর্শ নিন।"
    elif "collagen" in gen_low or "glucosamine" in gen_low:
        use_bn = "এটি হাঁটু ও জয়েন্টের ব্যথা (Osteoarthritis), তরুণাস্থি ক্ষয় রোধ এবং হাড়ের সন্ধির নমনীয়তা বাড়াতে ব্যবহৃত হয়।"
        rule_bn = "প্রতিদিন ১টি ট্যাবলেট ভরা পেটে (খাবারের পরে) এক গ্লাস পানির সাথে সেবন করুন।"
        warn_bn = "সতর্কতা: গর্ভাবস্থায়, স্তন্যদানকালে বা শেলফিশ অ্যালার্জি থাকলে চিকিৎসকের পরামর্শ নিয়ে সেবন করুন।"
    elif "vitamin b" in gen_low or "riboflavin" in gen_low or "thiamine" in gen_low:
        use_bn = "এটি শরীরে ভিটামিন-বি এর ঘাটতি পূরণ, স্নায়ুর দুর্বলতা, মুখে বা জিহ্বায় ঘা, রক্তস্বল্পতা এবং শারীরিক ক্লান্তি দূর করতে ব্যবহৃত হয়।"
        rule_bn = "প্রতিদিন ১ থেকে ২টি ট্যাবলেট বা ক্যাপসুল খাবারের পরে ভরা পেটে সেবন করা হয়।"
        warn_bn = "সতর্কতা: এটি সাধারণত নিরাপদ; তবে নির্ধারিত মাত্রার বেশি সেবন করবেন না।"
    elif any(k in cls_low or k in ind_low for k in ("antibiotic", "penicillin", "cephalosporin", "macrolide", "infection", "bacterial")):
        use_bn = f"এটি একটি অ্যান্টিবায়োটিক ওষুধ, যা ব্যাকটেরিয়াজনিত সংক্রমণ ({indication or 'Bacterial Infection'}) নিরাময়ে ব্যবহৃত হয়।"
        rule_bn = "চিকিৎসকের দেওয়া নির্দিষ্ট সময়ের ব্যবধানে (যেমন ৮ বা ১২ ঘণ্টা পরপর) পুরো কোর্স শেষ করতে হবে।"
        warn_bn = "সতর্কতা: সুস্থ বোধ করলেও অ্যান্টিবায়োটিকের পূর্ণ কোর্স মাঝপথে বন্ধ করবেন না।"
    else:
        ind_text = f" ({indication})" if indication else ""
        cls_text = f" [{drug_class}]" if drug_class else ""
        use_bn = f"এটি মূলত {ind_text or 'সংশ্লিষ্ট রোগের উপসর্গ'}{cls_text} নিরাময় ও নিয়ন্ত্রণে ব্যবহৃত হয়।"
        rule_bn = "প্রতিদিন চিকিৎসকের নির্ধারিত মাত্রায় খাবারের পরে পানি দিয়ে সেবন করুন।"
        warn_bn = "সতর্কতা: গর্ভাবস্থায়, স্তন্যদানকালে বা অন্য কোনো দীর্ঘমেয়াদী ওষুধের সাথে খাওয়ার আগে চিকিৎসকের পরামর্শ নিন।"

    # Override administration route if dosage form is NOT an oral tablet/capsule
    if is_suppository:
        rule_bn = "এটি একটি সাপোজিটরি, যা পায়ুপথে ব্যবহার করতে হয়: মুখে খাওয়া যাবে না। চিকিৎসকের নির্ধারিত মাত্রা ও সময়ের ব্যবধান অনুযায়ী ব্যবহার করুন।"
    elif is_topical:
        rule_bn = "এটি বাহ্যিক ব্যবহারের ওষুধ; আক্রান্ত স্থানে পরিষ্কার হাতে পাতলা করে লাগান: মুখে খাওয়া যাবে না।"
    elif is_drop:
        rule_bn = "এটি ড্রপ হিসেবে ব্যবহার্য; চিকিৎসকের নির্ধারিত ফোঁটা অনুযায়ী নির্দিষ্ট স্থানে প্রয়োগ করুন: মুখে খাওয়া যাবে না।"
    elif is_injection:
        rule_bn = "এটি একটি ইনজেকশন, যা শুধুমাত্র চিকিৎসক বা প্রশিক্ষিত নার্সের তত্ত্বাবধানে প্রয়োগ করতে হবে।"
    elif is_inhaler:
        rule_bn = "এটি শ্বাসের মাধ্যমে (ইনহেলার হিসেবে) গ্রহণ করতে হয়; চিকিৎসকের দেখানো নিয়ম অনুযায়ী ব্যবহার করুন।"
    elif is_liquid_oral:
        rule_bn = "ব্যবহারের আগে বোতল ভালোভাবে ঝাঁকিয়ে নিন এবং চিকিৎসকের নির্ধারিত মাপনি চামচ বা ড্রপার অনুযায়ী সেবন করুন।"

    return _format_bangla_summary(
        f"{brand_name}{str_part} হলো একটি {form_bn}{mfg_str}, যার মূল সক্রিয় উপাদান {generic}। "
        f"{use_bn} {rule_bn}\n{warn_bn}"
    )


class SafetySLMEngine:
    """
    Mini AI / Quantized Safety SLM component for context-aware OCR disambiguation
    and grounded Bangla medication schedule & safety reasoning.
    """

    def __init__(self, model_name: str = SLM_MODEL_NAME) -> None:
        self.model_name = model_name
        logger.info("[DEV-SLM] Initialized Quantized Safety SLM Engine (%s)", self.model_name)

    @staticmethod
    def filter_ghost_and_prefix_brands(
        brand_candidates: Set[str],
        all_ocr_words_lower: Set[str],
        full_ocr_text_lower: str,
        known_db_words: Optional[Set[str]] = None,
    ) -> Set[str]:
        refined = set(brand_candidates)

        # 1. Multi-word brand upgrade: if 'napa' is detected and 'extra' is in the OCR tokens, upgrade to 'napa extra'
        # Only upgrade when the combined brand is already in `refined` or exists in `known_db_words`.
        for base in list(refined):
            for mod in BRAND_MODIFIERS:
                if mod in all_ocr_words_lower:
                    combo = f"{base} {mod}"
                    if combo in refined or (
                        base == "napa"
                        and mod in ("extra", "extend", "rapid", "one")
                        and (known_db_words is None or combo in known_db_words)
                    ):
                        refined.add(combo)
                        if base in refined:
                            refined.discard(base)

        for b1 in list(refined):
            for b2 in list(refined):
                if b1 != b2 and b2.startswith(b1 + " "):
                    refined.discard(b1)

        # 2. Substring ghost suppression: only suppress `b` if the longer word `w` in the image is a genuine
        # dictionary word in `medicines.db` (e.g. 'amlodipine', 'diazepam', 'tufnil')
        valid_longer_pool = (
            {w for w in all_ocr_words_lower if w in known_db_words}
            if known_db_words
            else all_ocr_words_lower
        )
        for b in list(refined):
            if " " in b:
                continue
            is_substring_of_longer_word = any(
                len(w) >= len(b) + 2 and b in w
                for w in valid_longer_pool
            )
            if is_substring_of_longer_word:
                logger.info(
                    "[DEV-SLM] Suppressed torn-foil substring ghost brand %r (substring of longer valid drug word in image)",
                    b,
                )
                refined.discard(b)

        return refined

    @staticmethod
    def refine_strength_for_medicine(
        raw_cluster_text: str,
        all_detected_strengths: List[str],
        medicine: Optional[MedicineInfo],
        extract_strengths_fn,
    ) -> Tuple[Optional[str], List[str]]:
        if not medicine:
            primary = all_detected_strengths[0] if all_detected_strengths else None
            return primary, all_detected_strengths

        generic_lower = (medicine.generic or "").lower()
        brand_lower = (medicine.brand_name or "").lower()

        if "vitamin b" in generic_lower and "complex" in generic_lower:
            ingredient_parts = ["thiamine", "riboflavin", "pyridoxine", "nicotinamide"]
        else:
            ingredient_parts = [p.strip() for p in re.split(r"\+|/|&|,", generic_lower) if p.strip()]
        num_ingredients = max(1, len(ingredient_parts))

        if num_ingredients == 1:
            first_gen_word = ingredient_parts[0].split()[0] if ingredient_parts else ""
            weighted_counts: Counter[str] = Counter()

            for line in raw_cluster_text.splitlines():
                _, line_strengths = extract_strengths_fn(line, brand_name=medicine.brand_name)
                line_low = line.lower()
                weight = 1
                if brand_lower in line_low or (len(first_gen_word) >= 4 and first_gen_word in line_low):
                    weight = 4
                for st in line_strengths:
                    if "+" not in st:
                        weighted_counts[st] += weight

            if weighted_counts:
                best_single = weighted_counts.most_common(1)[0][0]
                return best_single, [best_single]

            single_only = [s for s in all_detected_strengths if "+" not in s]
            if single_only:
                return single_only[0], [single_only[0]]
            return (all_detected_strengths[0] if all_detected_strengths else None), all_detected_strengths

        ordered_strengths: List[str] = []
        lines = raw_cluster_text.splitlines()

        for ing in ingredient_parts:
            ing_words = [w for w in re.findall(r"[a-z]{4,}", ing) if w not in ("type", "acid", "sulfate")]
            matched_st_for_ing: Optional[str] = None
            for ln in lines:
                ln_low = ln.lower()
                if any(
                    w in ln_low or any(fuzz.ratio(w, tok) >= 75 for tok in re.findall(r"[a-z]{4,}", ln_low))
                    for w in ing_words
                ):
                    _, ln_sts = extract_strengths_fn(ln)
                    for st in ln_sts:
                        if "+" not in st and (st not in ordered_strengths or "vitamin b" in generic_lower):
                            matched_st_for_ing = st
                            break
                if matched_st_for_ing:
                    break
            if matched_st_for_ing:
                ordered_strengths.append(matched_st_for_ing)

        if "paracetamol" in generic_lower and "caffeine" in generic_lower:
            if "500 mg" in all_detected_strengths and "65 mg" in all_detected_strengths:
                return "500 mg + 65 mg", ["500 mg", "65 mg"]

        if len(ordered_strengths) >= 2:
            combined = " + ".join(ordered_strengths[:num_ingredients])
            return combined, ordered_strengths[:num_ingredients]

        singles = [s for s in all_detected_strengths if "+" not in s]
        if len(singles) >= 2:
            combined = " + ".join(singles[:num_ingredients])
            return combined, singles[:num_ingredients]

        primary = all_detected_strengths[0] if all_detected_strengths else None
        return primary, all_detected_strengths

    @staticmethod
    def build_schedule_and_guidance(
        strip: StripScanResult,
        is_duplicate: bool,
        is_interaction: bool,
    ) -> Tuple[str, str, str]:
        """
        Generates grounded Bangla dosage guidance (`dosageBangla`), Bangla beginner summary (`dosageEnglish`),
        and daily schedule timing tag (`timingTag`).
        """
        med = strip.matched_medicine
        if not med:
            return (
                "চিকিৎসকের পরামর্শ অনুযায়ী সেব্য",
                "চিকিৎসকের পরামর্শ অনুযায়ী সেব্য।",
                "চিকিৎসকের পরামর্শ অনুযায়ী",
            )

        strength_str = (strip.extracted_strength or med.package_container or "").strip()
        name_with_str = f"{med.brand_name} ({strength_str})" if strength_str else med.brand_name
        gen_low = (med.generic or "").lower()
        dosage_desc_low = (med.dosage_description or "").lower()
        form_low = (strip.detected_dosage_form or med.dosage_form or "tablet").strip().lower()

        bangla_summary = generate_bangla_beginner_summary(
            brand_name=med.brand_name,
            generic=med.generic,
            dosage_form=strip.detected_dosage_form or med.dosage_form,
            manufacturer=med.manufacturer,
            drug_class=med.drug_class,
            indication=med.indication,
            strength=strength_str,
            dosage_description=med.dosage_description,
            side_effects_description=med.side_effects_description,
            interaction_description=med.interaction_description,
            precautions_description=med.precautions_description or med.contraindications_description,
            use_llm=True,
        )

        if is_duplicate:
            return (
                f"⚠️ সতর্কতা: একই জেনেরিক উপাদান ({med.generic}) একাধিক পাতায় রয়েছে! ওভারডোজ এড়াতে চিকিৎসকের পরামর্শ ছাড়া একসাথে খাবেন না।",
                bangla_summary,
                "⚠️ বর্জন করুন (ডুপ্লিকেট)",
            )

        if is_interaction:
            return (
                f"⚠️ আন্তঃক্রিয়া সতর্কতা: {name_with_str} অন্য শনাক্তকৃত ওষুধের সাথে প্রতিক্রিয়া করতে পারে। একসাথে সেবনের আগে অবশ্যই চিকিৎসকের পরামর্শ নিন।",
                bangla_summary,
                "⚠️ সতর্কতা (আন্তঃক্রিয়া)",
            )

        # Non-oral / route-specific dosage forms must NEVER receive oral tablet/meal instructions
        if "suppository" in form_low:
            return (
                f"{name_with_str}: এটি একটি সাপোজিটরি (পায়ুপথে ব্যবহার্য), মুখে খাওয়া যাবে না। চিকিৎসকের নির্ধারিত মাত্রায় ব্যবহার করুন।",
                bangla_summary,
                "পায়ুপথে ব্যবহার্য (মুখে নয়)",
            )
        if any(k in form_low for k in ("cream", "ointment", "gel", "lotion", "shampoo", "spray", "paste")):
            return (
                f"{name_with_str}: বাহ্যিক ব্যবহারের ওষুধ; চিকিৎসকের পরামর্শ অনুযায়ী আক্রান্ত স্থানে লাগান (মুখে নয়)।",
                bangla_summary,
                "ত্বকে বাহ্যিক ব্যবহার্য",
            )
        if any(k in form_low for k in ("drop", "ophthalmic", "otic", "nasal")):
            return (
                f"{name_with_str}: ড্রপ হিসেবে ব্যবহার্য; চিকিৎসকের নির্ধারিত ফোঁটা অনুযায়ী প্রয়োগ করুন।",
                bangla_summary,
                "ড্রপ (চিকিৎসকের পরামর্শে)",
            )
        if "injection" in form_low or "infusion" in form_low:
            return (
                f"{name_with_str}: ইনজেকশন; শুধুমাত্র চিকিৎসক বা প্রশিক্ষিত নার্সের মাধ্যমে প্রয়োগ করতে হবে।",
                bangla_summary,
                "ইনজেকশন (চিকিৎসকের তত্ত্বাবধানে)",
            )
        if "inhaler" in form_low or "nebul" in form_low:
            return (
                f"{name_with_str}: ইনহেলার; চিকিৎসকের দেখানো নিয়ম অনুযায়ী শ্বাসের মাধ্যমে গ্রহণ করুন।",
                bangla_summary,
                "শ্বাসের মাধ্যমে (ইনহেলার)",
            )
        if any(k in form_low for k in ("syrup", "suspension", "elixir", "solution")):
            return (
                f"{name_with_str}: তরল ওষুধ; ভালোভাবে ঝাঁকিয়ে চিকিৎসকের নির্ধারিত মাপনি চামচ অনুযায়ী সেবন করুন।",
                bangla_summary,
                "মাপনি চামচে সেব্য (চিকিৎসকের পরামর্শে)",
            )

        if any(p in gen_low for p in ("omeprazole", "esomeprazole", "pantoprazole", "rabeprazole", "lansoprazole", "dexlansoprazole")):
            timing_tag = "সকাল • রাত (খাবারের ৩০ মি. আগে)"
            dosage_bn = f"{name_with_str}: ১টি করে দিনে ১ থেকে ২ বার, খাবারের ৩০ মিনিট আগে সেবন করুন।"
        elif "paracetamol" in gen_low:
            timing_tag = "সকাল • দুপুর • রাত (ভরা পেটে)"
            dosage_bn = f"{name_with_str}: প্রাপ্তবয়স্কদের জন্য ১টি ট্যাবলেট দিনে ৩ বার ভরা পেটে (অথবা জ্বর/ব্যথায় প্রয়োজনে)।"
        elif any(p in gen_low for p in ("naproxen", "ketorolac", "tolfenamic", "ibuprofen", "diclofenac", "aceclofenac", "etoricoxib")):
            timing_tag = "সকাল • রাত (অবশ্যই ভরা পেটে)"
            dosage_bn = f"{name_with_str}: ব্যথানাশক ওষুধ; গ্যাস্ট্রিকের ঝুঁকি এড়াতে অবশ্যই ভরা পেটে খাবারের পরে সেবন করুন।"
        elif any(p in gen_low for p in ("clonazepam", "diazepam", "alprazolam", "bromazepam", "flunarizine", "sertraline", "pregabalin")):
            timing_tag = "রাত (ঘুমানোর আগে / চিকিৎসকের পরামর্শে)"
            dosage_bn = f"{name_with_str}: স্নায়ু বা ঘুমের ওষুধ; তন্দ্রাভাব হতে পারে, তাই চিকিৎসকের নির্ধারিত মাত্রায় (সাধারণত রাতে) সেবন করুন।"
        elif any(p in gen_low for p in ("amlodipine", "losartan", "telmisartan", "bisoprolol", "atenolol", "olmesartan")):
            timing_tag = "সকাল (প্রতিদিন নির্দিষ্ট সময়ে)"
            dosage_bn = f"{name_with_str}: রক্তচাপ নিয়ন্ত্রণের ওষুধ; প্রতিদিন নির্দিষ্ট সময়ে ১টি করে নিয়মিত সেবন করুন।"
        elif "ondansetron" in gen_low or "domperidone" in gen_low:
            timing_tag = "খাবারের আগে (বমি ভাব হলে)"
            dosage_bn = f"{name_with_str}: বমি বা বমি ভাব প্রতিরোধে খাবারের আগে চিকিৎসকের পরামর্শ অনুযায়ী সেব্য।"
        elif "before meal" in dosage_desc_low or "before food" in dosage_desc_low:
            timing_tag = "খাবারের আগে সেব্য"
            dosage_bn = f"{name_with_str}: খাবারের আগে চিকিৎসকের পরামর্শ অনুযায়ী সেবন করুন।"
        else:
            timing_tag = "সকাল • রাত (ভরা পেটে)"
            dosage_bn = f"{name_with_str}: জেনেরিক: {med.generic}। খাবারের পরে চিকিৎসকের পরামর্শ অনুযায়ী সেবন করুন।"

        return dosage_bn, bangla_summary, timing_tag


# ---------------------------------------------------------------------------
# Caregiver Mode Grounded Follow-Up Q&A (Gemma 3 1B + Deterministic Bangla Fallback)
# ---------------------------------------------------------------------------

_CAREGIVER_QA_SYSTEM_PROMPT = (
    "তুমি OushudhCheck কেয়ারগিভার মেডিকেল সহায়ক। শুধুমাত্র প্রদত্ত [ডাটাবেস তথ্য] এবং "
    "[নিরাপত্তা সতর্কতা]-এর ভিত্তিতে কেয়ারগিভারের নির্দিষ্ট প্রশ্নের সরাসরি, স্পষ্ট ও সহজ বাংলায় "
    "৩-৫ বাক্যে উত্তর দাও। ওষুধের ব্র্যান্ড নাম, মাত্রা এবং জেনেরিক উপাদানের নাম ডাটাবেসে যেভাবে "
    "ইংরেজিতে আছে ঠিক সেভাবেই অপরিবর্তিত রাখবে, আর বাকি সম্পূর্ণ ব্যাখ্যা সহজ বাংলায় লিখবে। "
    "ডাটাবেসে নেই এমন কোনো মাত্রা, পার্শ্বপ্রতিক্রিয়া বা সতর্কতা মনগড়াভাবে যোগ করবে না। "
    "কোনো ওষুধ যদি মুখে খাওয়ার না হয় (যেমন সাপোজিটরি, ক্রিম, ড্রপ, ইনজেকশন, ইনহেলার), "
    "তবে কখনোই খালি পেটে, ভরা পেটে বা মুখে সেবনের কথা লিখবে না।"
)

_CAREGIVER_INTENT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "meal_timing": (
        "খাবারের আগে", "খাবারের পরে", "খাওয়ার আগে", "খাওয়ার পরে", "খাওয়ার আগে", "খাওয়ার পরে",
        "খালি পেটে", "ভরা পেটে", "পেট ভরা", "কখন খাব", "কখন খাবে", "কখন খাই", "কখন দিতে",
        "সকালে", "দুপুরে", "রাতে", "ঘুমানোর আগে", "খাবার", "সময়", "সময়",
        "before meal", "after meal", "before food", "after food", "empty stomach",
        "full stomach", "with food", "with meal", "when to take", "what time",
        "morning", "night", "bedtime", "timing", "meal",
        "khali pete", "vora pete", "bhora pete", "khabarer age", "khabarer pore", "kokhon",
    ),
    "dosage": (
        "কতবার", "কত বার", "কয়বার", "কয়বার", "কয়টা", "কয়টা", "কতটুকু", "কত দিন", "কতদিন",
        "কত ঘণ্টা", "কত ঘন্টা", "মাত্রা", "ডোজ", "নিয়ম", "নিয়ম", "কীভাবে", "কিভাবে", "কেমনে",
        "খাওয়ার নিয়ম", "খাওয়ার নিয়ম", "ব্যবহারের নিয়ম", "ব্যবহারবিধি", "সেবনবিধি", "দিনে",
        "dose", "dosage", "how to take", "how to use", "how many", "how much",
        "frequency", "twice", "thrice", "daily", "per day", "hours",
        "kotobar", "koybar", "kivabe", "niyom",
    ),
    "side_effects": (
        "পার্শ্বপ্রতিক্রিয়া", "পার্শ্ব প্রতিক্রিয়া", "পার্শ্বপ্রতিক্রিয়া", "পার্শ্ব প্রতিক্রিয়া",
        "সাইড ইফেক্ট", "ক্ষতি", "ক্ষতিকর", "ঘুম ঘুম", "ঘুম আসে", "মাথা ঘোরা", "মাথা ঘোরায়",
        "বমি", "অ্যালার্জি", "এলার্জি", "র‍্যাশ", "চুলকানি", "কোষ্ঠকাঠিন্য", "পাতলা পায়খানা",
        "পায়খানা", "পেট ব্যথা", "গ্যাস হয়",
        "side effect", "side-effect", "adverse", "drowsy", "drowsiness", "sleepy",
        "dizziness", "dizzy", "nausea", "vomit", "rash", "allergy", "diarrhea", "constipation",
    ),
    "pregnancy_lactation": (
        "গর্ভাবস্থা", "গর্ভাবস্থায়", "গর্ভাবস্থায়", "গর্ভবতী", "প্রেগন্যান্ট", "প্রেগনেন্ট",
        "প্রেগন্যান্সি", "বাচ্চা পেটে", "অন্তঃসত্ত্বা", "স্তন্যদান", "স্তন্যদানকালে",
        "বুকের দুধ", "দুধ খাওয়ানো", "দুধ খাওয়ানো", "দুধ পান", "শিশুকে দুধ", "প্রসূতি",
        "pregnant", "pregnancy", "lactation", "lactating", "breastfeed", "breast feed",
        "nursing", "milk", "fetus", "trimester", "gorvoboti", "gorvabostha",
    ),
    "interactions_duplicates": (
        "অন্য ওষুধ", "অন্য কোন ওষুধ", "অন্য কোনো ওষুধ", "একসাথে", "এক সাথে", "সাথে খাওয়া",
        "সাথে খাওয়া", "সাথে খেলে", "মিশিয়ে", "বিক্রিয়া", "বিক্রিয়া", "আন্তঃক্রিয়া", "আন্তঃক্রিয়া",
        "রিঅ্যাকশন", "রিয়েকশন", "ডুপ্লিকেট", "একই ওষুধ", "একই জেনেরিক", "ওভারডোজ",
        "interaction", "interact", "duplicate", "together", "combine", "conflict",
        "overdose", "another medicine", "other medicine", "other drug", "same generic",
        "eksathe", "onno oshudh",
    ),
    "storage": (
        "সংরক্ষণ", "কোথায় রাখব", "কোথায় রাখব", "কোথায় রাখবে", "কোথায় রাখবে",
        "কীভাবে রাখব", "কিভাবে রাখব", "ফ্রিজে", "ফ্রিজ", "রোদ", "তাপমাত্রা", "গরম",
        "মেয়াদ", "মেয়াদ", "রাখা যাবে",
        "store", "storage", "fridge", "refrigerator", "temperature", "sunlight",
        "heat", "moisture", "keep", "expiry", "expire", "rakbo", "rakhbo",
    ),
    "purpose": (
        "কী কাজ", "কি কাজ", "কাজ কী", "কাজ কি", "কীসের ওষুধ", "কিসের ওষুধ", "কেন খায়",
        "কেন খায়", "কেন দেয়", "কেন দেয়", "কী জন্য", "কি জন্য", "কোন রোগ", "উপকার",
        "কিসে কাজ", "কীসের জন্য", "কিসের জন্য",
        "what is this", "what does", "purpose", "indication", "treat", "used for",
        "why take", "what for", "ki kaj", "kiser oshudh", "keno",
    ),
    "warnings_precautions": (
        "সতর্কতা", "সাবধানতা", "সাবধান", "নিষেধ", "নিষেধাজ্ঞা", "কারা খেতে পারবে না",
        "কারা খাবে না", "খাওয়া যাবে না", "খাওয়া যাবে না", "বারণ", "ঝুঁকি", "নিরাপদ",
        "সমস্যা আছে", "কোনো সমস্যা", "কোন সমস্যা", "কিডনি", "লিভার", "হার্ট", "ডায়াবেটিস",
        "হাঁপানি", "অ্যাজমা", "আলসার", "অ্যালকোহল", "গাড়ি", "গাড়ি",
        "warning", "precaution", "contraindication", "avoid", "danger", "risk",
        "safe", "safety", "kidney", "liver", "heart", "asthma", "ulcer", "driving",
        "alcohol", "sotorkota",
    ),
}

_MULTI_STRIP_ALL_RE = re.compile(
    r"(?:একসাথে|এক\s*সাথে|দুইটা|দুটো|উভয়|উভয়|সবগুলো|সব\s*ওষুধ|both|together|all\s+of\s+them|combine)",
    re.IGNORECASE,
)


def _clean_db_snippet(text: Optional[str], max_len: int = 240) -> str:
    """Normalizes HTML/whitespace from medicines.db text fields and trims cleanly."""
    if not text:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", str(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    if len(cleaned) <= max_len:
        return cleaned
    return textwrap.shorten(cleaned, width=max_len, placeholder="...")


def _detect_caregiver_intents(question: str) -> List[str]:
    """Detects one or more caregiver question intents from Bangla, Banglish, and English text."""
    q_low = (question or "").strip().lower()
    if not q_low:
        return []
    matched: List[str] = []
    for intent, keywords in _CAREGIVER_INTENT_KEYWORDS.items():
        if any(kw in q_low for kw in keywords):
            matched.append(intent)
    return matched


def _strip_mentioned_in_question(question_low: str, q_ascii_words: List[str], strip: StripScanResult) -> bool:
    med = strip.matched_medicine
    if not med:
        return False

    candidates: List[str] = []
    brand_low = (med.brand_name or "").strip().lower()
    if brand_low:
        candidates.append(brand_low)
        brand_first = brand_low.split()[0]
        if len(brand_first) >= 3 and brand_first != brand_low:
            candidates.append(brand_first)

    generic_low = (med.generic or "").strip().lower()
    if generic_low:
        candidates.append(generic_low)
        for part in re.split(r"\+|/|&|,", generic_low):
            p = part.strip()
            if len(p) >= 4:
                candidates.append(p)
                first_w = p.split()[0]
                if len(first_w) >= 4:
                    candidates.append(first_w)

    for cand in candidates:
        if len(cand) < 3:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(cand)}(?![a-z0-9])", question_low):
            return True

    # Fuzzy match ASCII words >= 4 chars for minor voice-ASR spelling drift (e.g. 'seklo' -> 'seclo')
    if q_ascii_words:
        target_words = [c for c in candidates if len(c) >= 4 and " " not in c]
        for qw in q_ascii_words:
            if len(qw) < 4:
                continue
            for tw in target_words:
                if fuzz.ratio(qw, tw) >= 84:
                    return True
    return False


def _select_focused_strips(
    question: str,
    strips: Sequence[StripScanResult],
    intents: Sequence[str],
) -> Tuple[List[StripScanResult], List[StripScanResult]]:
    """
    Returns `(focused_strips, all_valid_strips)`.
    If multiple strips were scanned and `question` names a specific brand/generic, focuses on
    the named strip(s) unless the caregiver is explicitly asking about taking/comparing all strips together.
    """
    valid_strips = [s for s in strips if s is not None and s.matched_medicine is not None]
    if len(valid_strips) <= 1:
        return valid_strips, valid_strips

    q_low = (question or "").strip().lower()
    if _MULTI_STRIP_ALL_RE.search(q_low):
        return valid_strips, valid_strips

    q_ascii_words = re.findall(r"[a-z0-9]{3,}", q_low)
    named = [s for s in valid_strips if _strip_mentioned_in_question(q_low, q_ascii_words, s)]
    if named:
        # If asking about interactions/duplicates across scanned strips and only 1 strip was named,
        # keep all strips in view so cross-strip alerts are not hidden.
        if "interactions_duplicates" in intents and len(named) == 1:
            return valid_strips, valid_strips
        return named, valid_strips

    return valid_strips, valid_strips


def _filter_relevant_alerts(
    alerts: Sequence[SafetyAlert],
    focused_strips: Sequence[StripScanResult],
    total_valid_strips: int,
) -> List[SafetyAlert]:
    if not alerts:
        return []
    if len(focused_strips) >= total_valid_strips:
        return list(alerts)
    focused_indices = {s.strip_index for s in focused_strips}
    focused_brands = {
        (s.matched_medicine.brand_name or "").strip().lower()
        for s in focused_strips
        if s.matched_medicine
    }
    relevant: List[SafetyAlert] = []
    for alert in alerts:
        if any(idx in focused_indices for idx in (alert.involved_strips or [])):
            relevant.append(alert)
        elif any((b or "").strip().lower() in focused_brands for b in (alert.involved_brands or [])):
            relevant.append(alert)
    return relevant


def _strip_display_name(strip: StripScanResult) -> str:
    med = strip.matched_medicine
    if not med:
        return "অজানা ওষুধ"
    strength = (strip.extracted_strength or med.package_container or "").strip()
    return f"{med.brand_name} {strength}".strip() if strength else med.brand_name


def _extract_meal_timing_bn(med: MedicineInfo, dosage_form: str) -> str:
    """Returns route-safe Bangla meal/timing guidance grounded in dosage_form, generic, and dosage_description."""
    non_oral = _get_non_oral_route_rule(dosage_form)
    if non_oral:
        return non_oral

    form_low = (dosage_form or "tablet").strip().lower()
    gen_low = (med.generic or "").lower()
    dose_low = (med.dosage_description or "").lower()

    if any(k in gen_low for k in ("omeprazole", "esomeprazole", "pantoprazole", "rabeprazole", "lansoprazole", "dexlansoprazole")):
        return "এটি সাধারণত খাবারের ৩০ মিনিট আগে খালি পেটে পানি দিয়ে গিলে খেতে হয়।"
    if any(k in dose_low for k in ("before meal", "before food", "empty stomach", "30 minutes before", "30 min before", "1 hour before")):
        return "ডাটাবেস অনুযায়ী এটি সাধারণত খাবারের আগে (খালি পেটে) পানি দিয়ে সেবন করতে হয়।"
    if any(k in gen_low for k in ("naproxen", "tolfenamic", "ketorolac", "ibuprofen", "diclofenac", "aceclofenac", "etoricoxib", "meloxicam")):
        return "গ্যাস্ট্রিক বা পেটের জ্বালাপোড়া এড়াতে এটি অবশ্যই ভরা পেটে (খাবারের পরে বা খাবারের সাথে) সেবন করতে হবে।"
    if any(k in dose_low for k in ("after meal", "after food", "with food", "with meal", "full stomach")):
        return "ডাটাবেস অনুযায়ী এটি ভরা পেটে (খাবারের পরে বা খাবারের সাথে) সেবন করতে হয়।"
    if "paracetamol" in gen_low:
        return "এটি সাধারণত ভরা পেটে (খাবারের পরে) পানি দিয়ে সেবন করা হয়; প্রয়োজনে ৪ থেকে ৬ ঘণ্টা পরপর খাওয়া যায়।"
    if any(k in gen_low for k in ("clonazepam", "diazepam", "alprazolam", "bromazepam", "midazolam", "nitrazepam", "flunarizine")):
        return "এটি সাধারণত রাতে ঘুমানোর আগে অথবা চিকিৎসকের নির্ধারিত সময়ে সেবন করতে হয় (তন্দ্রাভাব হতে পারে)।"
    if any(k in gen_low for k in ("amlodipine", "losartan", "telmisartan", "bisoprolol", "atenolol", "olmesartan", "valsartan")):
        return "এটি প্রতিদিন নির্দিষ্ট সময়ে (সাধারণত সকালে) খাবারের আগে বা পরে নিয়মিত সেবন করা যায়।"
    if any(k in gen_low for k in ("ondansetron", "domperidone", "palonosetron")):
        return "এটি সাধারণত খাবারের ১৫ থেকে ৩০ মিনিট আগে অথবা বমি ভাব হলে চিকিৎসকের পরামর্শ অনুযায়ী সেবন করতে হয়।"
    if any(k in form_low for k in ("syrup", "suspension", "elixir", "solution")):
        return "ব্যবহারের আগে বোতল ভালোভাবে ঝাঁকিয়ে চিকিৎসকের নির্ধারিত মাপনি চামচ বা ড্রপার অনুযায়ী খাবারের পরে সেবন করুন।"
    return "এটি সাধারণত খাবারের পরে (ভরা পেটে) অথবা চিকিৎসকের নির্ধারিত সময় অনুযায়ী পানি দিয়ে সেবন করতে হয়।"


def _call_local_slm_caregiver_qa(
    question: str,
    focused_strips: Sequence[StripScanResult],
    relevant_alerts: Sequence[SafetyAlert],
    intents: Sequence[str],
) -> Optional[str]:
    """
    Calls local Gemma 3 1B (`SLM_API_URL`) for Caregiver Mode Q&A, strictly grounded in the
    scanned strip(s)' `matched_medicine` fields and `alerts`, respecting `_SLM_CIRCUIT_OPEN_UNTIL`,
    `SLM_TIMEOUT_SEC`, and `SLM_MAX_TOKENS`.
    """
    global _SLM_CIRCUIT_OPEN_UNTIL
    if not SLM_API_URL or not focused_strips:
        return None

    strip_sig = tuple(
        (
            s.matched_medicine.brand_id if s.matched_medicine else 0,
            s.extracted_strength or "",
            s.detected_dosage_form or (s.matched_medicine.dosage_form if s.matched_medicine else "") or "",
        )
        for s in focused_strips
    )
    alert_sig = tuple((a.alert_type, a.generic_or_class) for a in relevant_alerts)
    cache_key = (
        re.sub(r"\s+", " ", (question or "").strip().lower()),
        json.dumps(strip_sig, ensure_ascii=False),
        json.dumps(alert_sig, ensure_ascii=False),
    )

    with _SLM_CACHE_LOCK:
        cached = _SLM_QA_CACHE.get(cache_key)
        if cached is not None:
            _SLM_QA_CACHE.move_to_end(cache_key)
            return cached
        if time.monotonic() < _SLM_CIRCUIT_OPEN_UNTIL:
            return None

    # Build grounded prompt blocks for each focused strip
    per_strip_budget = 1.0 if len(focused_strips) == 1 else 0.65
    lines: List[str] = ["[ডাটাবেস তথ্য]"]
    non_oral_entries: List[Tuple[str, str, str]] = []

    for idx, strip in enumerate(focused_strips, start=1):
        med = strip.matched_medicine
        if not med:
            continue
        dosage_form = (strip.detected_dosage_form or med.dosage_form or "Tablet").strip()
        strength = (strip.extracted_strength or med.package_container or "").strip()
        non_oral_rule = _get_non_oral_route_rule(dosage_form)
        if non_oral_rule:
            non_oral_entries.append((med.brand_name, dosage_form, non_oral_rule))

        if len(focused_strips) > 1:
            lines.append(f"--- ওষুধ {idx} ---")

        def _lim(base: int, boost: bool = False) -> int:
            return max(80, int((base + (140 if boost else 0)) * per_strip_budget))

        fields = [
            ("ওষুধের নাম", med.brand_name, 120),
            ("মাত্রা", strength, 120),
            ("ধরন", dosage_form, 80),
            ("জেনেরিক উপাদান", med.generic, 200),
            ("প্রস্তুতকারক", med.manufacturer, 100),
            ("ওষুধের শ্রেণি", med.drug_class, 140),
            ("ব্যবহারের নির্দেশনা", med.indication, _lim(220, "purpose" in intents)),
            ("বিস্তারিত নির্দেশনা", med.indication_description, _lim(260, "purpose" in intents)),
            ("সেবনবিধি ও মাত্রা", med.dosage_description, _lim(320, "meal_timing" in intents or "dosage" in intents)),
            ("পার্শ্বপ্রতিক্রিয়া", med.side_effects_description, _lim(260, "side_effects" in intents)),
            ("অন্য ওষুধের সাথে বিক্রিয়া", med.interaction_description, _lim(260, "interactions_duplicates" in intents)),
            ("গর্ভাবস্থা ও স্তন্যদান", med.pregnancy_and_lactation_description, _lim(260, "pregnancy_lactation" in intents)),
            ("সতর্কতা", med.precautions_description, _lim(240, "warnings_precautions" in intents)),
            ("নিষেধাজ্ঞা", med.contraindications_description, _lim(220, "warnings_precautions" in intents)),
            ("সংরক্ষণ", med.storage_conditions_description, _lim(180, "storage" in intents)),
        ]
        for label, val, limit in fields:
            clean = _clean_db_snippet(val, max_len=limit)
            if clean:
                lines.append(f"{label}: {clean}")
        if non_oral_rule:
            lines.append(
                f"ব্যবহারের মাধ্যম (গুরুত্বপূর্ণ): {non_oral_rule} "
                "কখনোই খালি পেটে, ভরা পেটে, খাবারের আগে/পরে বা মুখে সেবন লিখবে না।"
            )

    if relevant_alerts:
        lines.append("\n[নিরাপত্তা সতর্কতা]")
        for alert in relevant_alerts:
            msg = (alert.message_bn or alert.message_en or "").strip()
            if msg:
                lines.append(f"- ({alert.alert_type}, {alert.severity}): {msg}")

    q_clean = (question or "").strip() or "এই ওষুধটির কাজ, ব্যবহারের নিয়ম এবং সতর্কতা কী?"
    lines.append(f"\nকেয়ারগিভারের প্রশ্ন: {q_clean}")
    user_prompt = "\n".join(lines)

    payload = {
        "model": SLM_MODEL_NAME,
        "messages": [
            {"role": "user", "content": f"{_CAREGIVER_QA_SYSTEM_PROMPT}\n\n{user_prompt}"},
        ],
        "temperature": 0.15,
        "top_p": 0.9,
        "repeat_penalty": 1.05,
        "max_tokens": SLM_MAX_TOKENS,
        "stream": False,
    }

    req = urllib.request.Request(
        SLM_API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=SLM_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        if content and re.search(r"[\u0980-\u09FF]{10,}", content):
            formatted = _format_bangla_summary(content)
            # Only apply single-form `_enforce_route_safety` when ALL focused strips share the SAME non-oral route;
            # for mixed oral + non-oral or multiple distinct non-oral routes, append an explicit per-strip warning
            # for EVERY non-oral strip so no strip is skipped or given the wrong route.
            if (
                non_oral_entries
                and len(non_oral_entries) == len(focused_strips)
                and len({rule for _, _, rule in non_oral_entries}) == 1
            ):
                formatted = _enforce_route_safety(formatted, non_oral_entries[0][1])
            elif non_oral_entries:
                extra_warnings: List[str] = []
                for b_name, d_form, rule in non_oral_entries:
                    if rule not in formatted:
                        extra_warnings.append(f"{b_name} ({d_form}): {rule}")
                if extra_warnings:
                    formatted = _format_bangla_summary(
                        f"{formatted}\nসতর্কতা: " + " ".join(extra_warnings)
                    )

            if not _is_slm_qa_answer_clinically_valid(formatted, focused_strips, relevant_alerts, intents):
                logger.info(
                    "[DEV-SLM] Caregiver Q&A SLM output failed clinical/intent grounding check; using deterministic fallback."
                )
                return None

            with _SLM_CACHE_LOCK:
                _SLM_QA_CACHE[cache_key] = formatted
                _SLM_QA_CACHE.move_to_end(cache_key)
                while len(_SLM_QA_CACHE) > _SLM_QA_CACHE_MAX:
                    _SLM_QA_CACHE.popitem(last=False)
            return formatted
        return None
    except Exception as exc:
        with _SLM_CACHE_LOCK:
            _SLM_CIRCUIT_OPEN_UNTIL = time.monotonic() + _SLM_CIRCUIT_COOLDOWN_SEC
        logger.warning(
            "[DEV-SLM] Local SLM unavailable for caregiver Q&A (%s); using grounded Bangla fallback (cooldown %.0fs).",
            exc,
            _SLM_CIRCUIT_COOLDOWN_SEC,
        )
        return None


def _is_slm_qa_answer_clinically_valid(
    answer: str,
    focused_strips: Sequence[StripScanResult],
    relevant_alerts: Sequence[SafetyAlert],
    intents: Sequence[str],
) -> bool:
    """
    Post-generation clinical grounding validator for 1B SLM caregiver Q&A output.
    Rejects meal-timing contradictions (e.g. telling a patient to take Omeprazole after meal),
    missing multi-strip coverage, or off-topic summary regressions so the deterministic
    grounded Bangla engine answers safely instead.
    """
    if not answer or not focused_strips:
        return False
    ans_low = answer.lower()

    # 1. Multi-strip coverage: every focused strip's brand or generic should be addressed
    if len(focused_strips) > 1:
        for strip in focused_strips:
            med = strip.matched_medicine
            if not med:
                continue
            brand_tok = (med.brand_name or "").strip().lower().split()[0] if med.brand_name else ""
            gen_tok = (med.generic or "").strip().lower().split()[0] if med.generic else ""
            if brand_tok and brand_tok not in ans_low and (not gen_tok or gen_tok not in ans_low):
                return False

    # 2. Single-strip meal-timing contradiction guard (e.g. PPI before-meal vs NSAID after-meal)
    if len(focused_strips) == 1 and focused_strips[0].matched_medicine:
        med = focused_strips[0].matched_medicine
        d_form = (focused_strips[0].detected_dosage_form or med.dosage_form or "Tablet").strip()
        expected_timing = _extract_meal_timing_bn(med, d_form)
        if ("খাবারের ৩০ মিনিট আগে" in expected_timing or "খাবারের আগে" in expected_timing) and (
            "খাবারের পরে" in answer or "ভরা পেটে" in answer
        ):
            return False
        if "অবশ্যই ভরা পেটে" in expected_timing and ("খালি পেটে" in answer or "খাবারের আগে" in answer):
            return False

    # 3. Intent relevance guard: ensure the 1B model actually addressed the asked topic
    if "storage" in intents:
        if not any(k in ans_low for k in ("সংরক্ষণ", "তাপমাত্রা", "শুষ্ক", "ঠান্ডা", "রোদ", "ফ্রিজ", "আলো", "আর্দ্রতা", "°", "30 c", "25 c")):
            return False
    if "pregnancy_lactation" in intents:
        if not any(k in ans_low for k in ("গর্ভাবস্থা", "গর্ভবতী", "স্তন্যদান", "বুকের দুধ", "প্রসূতি", "pregnancy", "lactation")):
            return False
    if "interactions_duplicates" in intents and relevant_alerts:
        if not any(k in ans_low for k in ("ডুপ্লিকেট", "একই জেনেরিক", "বিক্রিয়া", "আন্তঃক্রিয়া", "একসাথে")):
            return False
    if "side_effects" in intents:
        if not any(k in ans_low for k in ("পার্শ্বপ্রতিক্রিয়া", "পার্শ্ব", "বমি", "মাথাব্যথা", "মাথা ঘোরা", "ঘুম", "অ্যালার্জি", "র‍্যাশ", "পেট", "ক্ষতি")):
            return False

    return True


_ORAL_DOSAGE_DESC_RE = re.compile(
    r"\b(?:tablet|tablets|capsule|capsules|syrup|oral|mouth|swallow)\b",
    re.IGNORECASE,
)


def _extract_form_safe_dosage_snippet(
    dosage_description: Optional[str],
    dosage_form: str,
    max_len: int = 240,
) -> str:
    """
    Returns a clean `dosage_description` snippet safe for the given `dosage_form`.
    For non-oral forms (Suppository, Cream, Drops, Injection, Inhaler), prevents shared-generic
    oral `Tablet:` / `Capsule:` dosage instructions from leaking onto non-oral strips by
    extracting only a matching non-oral clause (or returning '' if none exists).
    """
    if not dosage_description:
        return ""
    if not _get_non_oral_route_rule(dosage_form):
        return _clean_db_snippet(dosage_description, max_len=max_len)

    full = re.sub(r"<[^>]+>", " ", str(dosage_description))
    full = re.sub(r"\s+", " ", full).strip()
    if not full:
        return ""

    form_low = (dosage_form or "").strip().lower()
    if "suppository" in form_low:
        form_kws = ("suppository", "suppositories", "rectal", "rectally")
    elif any(k in form_low for k in ("cream", "ointment", "gel", "lotion", "shampoo", "spray", "paste")):
        form_kws = ("cream", "ointment", "gel", "lotion", "topical", "apply", "spray", "paste", "shampoo")
    elif any(k in form_low for k in ("drop", "ophthalmic", "otic", "nasal")):
        form_kws = ("drop", "drops", "ophthalmic", "eye", "ear", "otic", "nasal", "instill")
    elif "injection" in form_low or "infusion" in form_low:
        form_kws = ("injection", "injectable", "infusion", "intravenous", "intramuscular", "iv", "im", "subcutaneous")
    elif "inhaler" in form_low or "nebul" in form_low:
        form_kws = ("inhaler", "inhalation", "puff", "puffs", "nebul")
    else:
        form_kws = ()

    # Split into sentences/sections and collect only clauses matching this non-oral route
    # and not describing oral tablet/capsule administration.
    clauses = [c.strip() for c in re.split(r"(?<=[.;।])\s+|\n+", full) if c.strip()]
    matched_clauses: List[str] = []
    for clause in clauses:
        clause_low = clause.lower()
        if any(re.search(rf"\b{re.escape(kw)}\b", clause_low) for kw in form_kws):
            if not _ORAL_DOSAGE_DESC_RE.search(clause):
                matched_clauses.append(clause)
    if not matched_clauses:
        return ""
    return _clean_db_snippet(" ".join(matched_clauses), max_len=max_len)


def _build_deterministic_caregiver_answer(
    question: str,
    focused_strips: Sequence[StripScanResult],
    relevant_alerts: Sequence[SafetyAlert],
    intents: Sequence[str],
) -> str:
    """
    Bulletproof deterministic Bangla Q&A fallback grounded strictly in `medicines.db` fields
    and safety alerts when local Gemma 3 1B is offline or unavailable.
    """
    sections: List[str] = []

    # Lead with safety alerts when asking about interactions/duplicates or warnings, or when high-severity alerts exist
    show_alerts_first = bool(relevant_alerts) and (
        "interactions_duplicates" in intents
        or "warnings_precautions" in intents
        or any((a.severity or "").lower() == "high" for a in relevant_alerts)
    )
    if show_alerts_first:
        alert_lines = ["⚠️ নিরাপত্তা সতর্কতা:"]
        for alert in relevant_alerts:
            msg = (alert.message_bn or alert.message_en or "").strip()
            if msg:
                alert_lines.append(f"• {msg}")
        if len(alert_lines) > 1:
            sections.append("\n".join(alert_lines))

    # No specific intent matched -> grounded summary of each focused strip
    if not intents:
        for strip in focused_strips:
            med = strip.matched_medicine
            if not med:
                continue
            d_form = (strip.detected_dosage_form or med.dosage_form or "Tablet").strip()
            strength = (strip.extracted_strength or med.package_container or "").strip()
            summary = generate_bangla_beginner_summary(
                brand_name=med.brand_name,
                generic=med.generic,
                dosage_form=d_form,
                manufacturer=med.manufacturer,
                drug_class=med.drug_class,
                indication=med.indication,
                strength=strength,
                dosage_description=med.dosage_description,
                side_effects_description=med.side_effects_description,
                interaction_description=med.interaction_description,
                precautions_description=med.precautions_description or med.contraindications_description,
                use_llm=False,
            )
            summary = _enforce_route_safety(summary, d_form)
            sections.append(summary)
        return _format_bangla_summary("\n\n".join(sections))

    for strip in focused_strips:
        med = strip.matched_medicine
        if not med:
            continue
        d_form = (strip.detected_dosage_form or med.dosage_form or "Tablet").strip()
        form_bn = _DOSAGE_FORM_BN.get(d_form.lower(), d_form)
        name_str = _strip_display_name(strip)
        header = f"{name_str} ({med.generic}, {form_bn}):"
        non_oral_rule = _get_non_oral_route_rule(d_form)
        parts: List[str] = []

        if "meal_timing" in intents or "dosage" in intents:
            timing_bn = _extract_meal_timing_bn(med, d_form)
            parts.append(timing_bn)
            dose_snip = _extract_form_safe_dosage_snippet(med.dosage_description, d_form, max_len=240)
            if dose_snip:
                parts.append(f"ডাটাবেস অনুযায়ী সেবনবিধি/মাত্রা: {dose_snip}")
            if "dosage" in intents:
                parts.append("চিকিৎসকের নির্ধারিত মাত্রা ও সময়সীমা মেনে ব্যবহার করুন।")

        if "purpose" in intents:
            ind_main = _clean_db_snippet(med.indication, max_len=140)
            ind_desc = _clean_db_snippet(med.indication_description, max_len=240)
            cls_txt = f" ({med.drug_class})" if med.drug_class else ""
            if ind_desc:
                parts.append(f"এটি{cls_txt} মূলত {ind_main or 'সংশ্লিষ্ট উপসর্গ'} নিরাময়ে ব্যবহৃত হয়। ডাটাবেস তথ্য: {ind_desc}")
            elif ind_main:
                parts.append(f"এটি{cls_txt} মূলত {ind_main} নিরাময় ও নিয়ন্ত্রণে ব্যবহৃত হয়।")
            else:
                parts.append(f"এটি {med.generic}{cls_txt} উপাদানযুক্ত ওষুধ; নির্দিষ্ট রোগের জন্য চিকিৎসকের পরামর্শে ব্যবহৃত হয়।")

        if "side_effects" in intents:
            se_snip = _clean_db_snippet(med.side_effects_description, max_len=260)
            if se_snip:
                parts.append(f"ডাটাবেস অনুযায়ী সম্ভাব্য পার্শ্বপ্রতিক্রিয়া: {se_snip}")
            else:
                parts.append(
                    f"{med.brand_name} ({med.generic})-এর নির্দিষ্ট পার্শ্বপ্রতিক্রিয়ার তালিকা ডাটাবেসে উল্লেখ নেই; "
                    "তবে অ্যালার্জি, তীব্র অস্বস্তি বা অস্বাভাবিক কোনো লক্ষণ দেখা দিলে দ্রুত চিকিৎসকের পরামর্শ নিন।"
                )

        if "pregnancy_lactation" in intents:
            preg_snip = _clean_db_snippet(med.pregnancy_and_lactation_description, max_len=260)
            if preg_snip:
                parts.append(f"গর্ভাবস্থা ও স্তন্যদানকালে (ডাটাবেস তথ্য): {preg_snip}")
                parts.append("গর্ভাবস্থায় বা শিশুকে বুকের দুধ খাওয়ানোর সময় এই ওষুধ ব্যবহারের আগে অবশ্যই চিকিৎসকের অনুমতি নিন।")
            else:
                parts.append(
                    f"ডাটাবেসে {med.brand_name} ({med.generic})-এর গর্ভাবস্থা বা স্তন্যদান সংক্রান্ত নির্দিষ্ট তথ্য পাওয়া যায়নি। "
                    "নিরাপত্তার জন্য চিকিৎসকের পরামর্শ ছাড়া গর্ভাবস্থায় বা স্তন্যদানকালে এটি গ্রহণ করবেন না।"
                )

        if "interactions_duplicates" in intents:
            inter_snip = _clean_db_snippet(med.interaction_description, max_len=260)
            if inter_snip:
                parts.append(f"অন্য ওষুধের সাথে বিক্রিয়া (ডাটাবেস তথ্য): {inter_snip}")
            elif not relevant_alerts:
                parts.append(
                    f"{med.brand_name} ({med.generic})-এর জন্য স্ক্যানকৃত ওষুধগুলোর মধ্যে কোনো স্বয়ংক্রিয় ডুপ্লিকেট বা গুরুতর বিক্রিয়া শনাক্ত হয়নি; "
                    "তবে অন্য কোনো নিয়মিত ওষুধের সাথে একসাথে নেওয়ার আগে চিকিৎসক বা ফার্মাসিস্টের পরামর্শ নিন।"
                )

        if "storage" in intents:
            stor_snip = _clean_db_snippet(med.storage_conditions_description, max_len=220)
            if stor_snip:
                parts.append(f"সংরক্ষণের নিয়ম (ডাটাবেস তথ্য): {stor_snip}")
            else:
                parts.append("শুষ্ক ও ঠান্ডা স্থানে, সরাসরি রোদ ও আর্দ্রতা থেকে দূরে এবং শিশুদের নাগালের বাইরে সংরক্ষণ করুন।")

        if "warnings_precautions" in intents:
            prec_snip = _clean_db_snippet(med.precautions_description, max_len=220)
            contra_snip = _clean_db_snippet(med.contraindications_description, max_len=200)
            if non_oral_rule and non_oral_rule not in parts:
                parts.append(non_oral_rule)
            if prec_snip:
                parts.append(f"সতর্কতা (ডাটাবেস তথ্য): {prec_snip}")
            if contra_snip:
                parts.append(f"নিষেধাজ্ঞা: {contra_snip}")
            if not prec_snip and not contra_snip:
                parts.append(
                    "সতর্কতা: দীর্ঘমেয়াদী কোনো রোগ (যেমন কিডনি, লিভার বা হৃদরোগ), অ্যালার্জি বা অন্য ওষুধ চলমান থাকলে "
                    "ব্যবহারের আগে অবশ্যই চিকিৎসকের পরামর্শ নিন।"
                )

        if not parts:
            continue

        # Ensure non-oral warning is always respected and oral phrasing can never leak
        block = f"{header} " + " ".join(parts)
        if non_oral_rule:
            block = _enforce_route_safety(block, d_form)
        sections.append(block)

    # Append a concise closing safety note if not already ending with a 'সতর্কতা:' line
    combined = "\n\n".join(sections).strip()
    if "সতর্কতা:" not in combined and "⚠️" not in combined:
        combined = f"{combined}\nসতর্কতা: যেকোনো নতুন উপসর্গ বা মাত্রা পরিবর্তনের আগে চিকিৎসকের পরামর্শ নিন।"
    return _format_bangla_summary(combined)


def answer_caregiver_followup(
    question: str,
    strips: Sequence[StripScanResult],
    alerts: Sequence[SafetyAlert] = (),
) -> str:
    """
    Answers a caregiver's voice or text follow-up question in clear Bangla grounded strictly in
    the scanned medicine strip(s)' `matched_medicine` fields and any `SafetyAlert` objects.

    Uses local Gemma 3 1B (`SLM_API_URL`) when available (with route-safety enforcement for
    non-oral dosage forms), and automatically falls back to a deterministic grounded Bangla
    Q&A engine when the local SLM server is offline or unavailable.
    """
    global _LAST_SLM_USED_LIVE

    valid_strips = [s for s in (strips or ()) if s is not None and s.matched_medicine is not None]
    if not valid_strips:
        return (
            "কোনো শনাক্তকৃত ওষুধের তথ্য পাওয়া যায়নি। "
            "অনুগ্রহ করে আগে ওষুধের পাতার একটি স্পষ্ট ছবি পাঠান, তারপর আপনার প্রশ্নটি করুন।"
        )

    q_clean = (question or "").strip()
    intents = _detect_caregiver_intents(q_clean)
    focused_strips, all_valid = _select_focused_strips(q_clean, valid_strips, intents)
    relevant_alerts = _filter_relevant_alerts(alerts or (), focused_strips, len(all_valid))

    if SLM_API_URL:
        llm_answer = _call_local_slm_caregiver_qa(
            question=q_clean,
            focused_strips=focused_strips,
            relevant_alerts=relevant_alerts,
            intents=intents,
        )
        if llm_answer:
            with _SLM_CACHE_LOCK:
                _LAST_SLM_USED_LIVE = True
            return llm_answer
        with _SLM_CACHE_LOCK:
            _LAST_SLM_USED_LIVE = False

    return _build_deterministic_caregiver_answer(
        question=q_clean,
        focused_strips=focused_strips,
        relevant_alerts=relevant_alerts,
        intents=intents,
    )
