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
    from backend.app.models.schemas import MedicineInfo, StripScanResult
except ImportError:
    from app.core.config import PROJECT_ROOT  # type: ignore
    from app.models.schemas import MedicineInfo, StripScanResult  # type: ignore

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
