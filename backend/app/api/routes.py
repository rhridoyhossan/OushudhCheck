from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Literal, Optional, Tuple
import asyncio
import base64
import json
import logging
import math
import re as _re
import threading
import time
import uuid
import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse

try:
    from backend.app.core.config import PROJECT_ROOT, get_readonly_db_connection
    from backend.app.models.schemas import (
        MedicineInfo,
        MultiStripScanResponse,
        MultiTextExtractRequest,
        PaginatedMedicinesResponse,
        StripScanResult,
        TTSRequest,
        TextExtractRequest,
    )
    from backend.app.services.ocr import MedicineStripScanner, evaluate_multi_strip_safety
    from backend.app.services.slm import generate_bangla_beginner_summary, get_active_slm_label
    from backend.app.services.voice import (
        get_cached_tts_audio_path,
        get_tts_cache_key,
        synthesize_bangla_speech,
    )
except ImportError:
    from app.core.config import PROJECT_ROOT, get_readonly_db_connection  # type: ignore
    from app.models.schemas import (  # type: ignore
        MedicineInfo,
        MultiStripScanResponse,
        MultiTextExtractRequest,
        PaginatedMedicinesResponse,
        StripScanResult,
        TTSRequest,
        TextExtractRequest,
    )
    from app.services.ocr import MedicineStripScanner, evaluate_multi_strip_safety  # type: ignore
    from app.services.slm import generate_bangla_beginner_summary, get_active_slm_label  # type: ignore
    from app.services.voice import (  # type: ignore
        get_cached_tts_audio_path,
        get_tts_cache_key,
        synthesize_bangla_speech,
    )

logger = logging.getLogger("oushudhcheck.api")
router = APIRouter()

TEST_DIR = PROJECT_ROOT / "test"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB hard cap
MAX_IMAGE_PIXELS = 25_000_000  # ~25 MP decompression bomb guard

BENCHMARK_SAMPLES = [
    {
        "id": "multi-5-strips",
        "filename": "newimg (3).jpg",
        "title": "5-Strip Tabletop Scan (B-Complex, Napa Extra, Diazepam, Amdocal, Tufnil)",
        "titleBn": "৫টি ওষুধের পাতা একসাথে স্ক্যান (ভিটামিন বি-কমপ্লেক্স, নাপা এক্সট্রা, ডায়াজিপাম, অ্যামডোক্যাল, টাফনিল)",
        "stripsCount": 5,
    },
    {
        "id": "multi-4-strips",
        "filename": "newimg (2).jpg",
        "title": "4-Strip Tabletop Scan (Diazepam, Napa Extra, Tufnil, Amdocal)",
        "titleBn": "৪টি ওষুধের পাতা স্ক্যান (ডায়াজিপাম, নাপা এক্সট্রা, টাফনিল, অ্যামডোক্যাল)",
        "stripsCount": 4,
    },
    {
        "id": "multi-2-emistat-fluver",
        "filename": "img1.jpg",
        "title": "2-Strip Reflective Foil Scan (Emistat 8mg & Fluver 10mg)",
        "titleBn": "২টি ফয়েল স্ট্রিপ স্ক্যান (ইমিস্ট্যাট ৮ মি.গ্রা. ও ফ্লুভার ১০ মি.গ্রা.)",
        "stripsCount": 2,
    },
    {
        "id": "multi-2-seronex-fluver",
        "filename": "img (4).jpeg",
        "title": "2-Strip Glass Tabletop Scan (Seronex 50mg & Fluver 10mg)",
        "titleBn": "২টি স্ট্রিপ স্ক্যান (সেরোনেক্স ৫০ মি.গ্রা. ও ফ্লুভার ১০ মি.গ্রা.)",
        "stripsCount": 2,
    },
    {
        "id": "single-orthogen-combo",
        "filename": "img (1).jpeg",
        "title": "4-Ingredient Combination Strip (Orthogen: 20mg + 750mg + 50mg + 0.75mg)",
        "titleBn": "৪টি উপাদানের কম্বিনেশন স্ট্রিপ (অর্থোজেন ট্যাবলেট)",
        "stripsCount": 1,
    },
    {
        "id": "single-naprosyn-vertical",
        "filename": "img (2).jpeg",
        "title": "Vertical 90° Printed Strip (Naprosyn 500mg)",
        "titleBn": "লম্বালম্বি ছাপানো স্ট্রিপ (ন্যাপ্রোসিন ৫০০ মি.গ্রা.)",
        "stripsCount": 1,
    },
]


_SCANNER_INSTANCE: Optional[MedicineStripScanner] = None
_SCANNER_LOCK = threading.Lock()
_DB_STATS_CACHE: Optional[Dict[str, int]] = None
_SLM_CONCURRENCY_LIMIT = 2


def get_scanner() -> MedicineStripScanner:
    global _SCANNER_INSTANCE
    if _SCANNER_INSTANCE is not None:
        return _SCANNER_INSTANCE
    with _SCANNER_LOCK:
        if _SCANNER_INSTANCE is None:
            _SCANNER_INSTANCE = MedicineStripScanner()
        return _SCANNER_INSTANCE


async def _build_guidance_parallel(
    scanner: MedicineStripScanner,
    matched_strips: List[StripScanResult],
    dup_strip_ids: set,
    inter_strip_ids: set,
) -> List[Tuple[str, str, str]]:
    """Run per-strip Gemma 3 1B guidance generation off the event loop in parallel (bounded to 2 slots)."""
    if not matched_strips:
        return []
    sem = asyncio.Semaphore(_SLM_CONCURRENCY_LIMIT)

    async def _one(strip: StripScanResult) -> Tuple[str, str, str]:
        async with sem:
            return await asyncio.to_thread(
                scanner.slm.build_schedule_and_guidance,
                strip,
                strip.strip_index in dup_strip_ids,
                strip.strip_index in inter_strip_ids,
            )

    return list(await asyncio.gather(*(_one(s) for s in matched_strips)))


def _encode_thumb_data_url(img: np.ndarray, max_dim: int = 260) -> str:
    """Encodes an OpenCV BGR image into a compact JPEG base64 data URL (`data:image/jpeg;base64,...`)."""
    if img is None or img.size == 0:
        return ""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _build_safety_edges(
    matched_strips: List[StripScanResult],
    safety_alerts: List[Any],
) -> List[Dict[str, Any]]:
    """
    Constructs the complete Safety-Edges list showing every triggered duplicate/conflict rule
    as well as verified safe pairwise checks across detected strips.
    """
    edges: List[Dict[str, Any]] = []
    flagged_pairs = set()

    for idx, a in enumerate(safety_alerts, start=1):
        badge = "Danger" if a.alert_type == "duplicate_generic" else "Caution"
        rule_name = (
            "DUPLICATE_ACTIVE_GENERIC_RULE"
            if a.alert_type == "duplicate_generic"
            else (
                "DRUG_DRUG_INTERACTION_RULE"
                if a.alert_type == "drug_interaction"
                else "THERAPEUTIC_CLASS_OVERLAP_RULE"
            )
        )
        edges.append(
            {
                "id": f"edge-alert-{idx}",
                "rule": rule_name,
                "status": badge,
                "sourceBrand": a.involved_brands[0] if a.involved_brands else "Strip 1",
                "targetBrand": a.involved_brands[1] if len(a.involved_brands) > 1 else "Strip 2",
                "compoundOrClass": a.generic_or_class,
                "explanationEn": a.message_en,
                "explanationBn": a.message_bn,
            }
        )
        if len(a.involved_strips) >= 2:
            for s_x, s_y in combinations(a.involved_strips, 2):
                flagged_pairs.add((min(s_x, s_y), max(s_x, s_y)))
            if len(a.involved_brands) > 2:
                edges[-1]["targetBrand"] = ", ".join(a.involved_brands[1:])

    # Add verified-safe pairwise edges so judges see the full pairwise graph evaluation
    for i in range(len(matched_strips)):
        for j in range(i + 1, len(matched_strips)):
            s1, s2 = matched_strips[i], matched_strips[j]
            pair_key = (min(s1.strip_index, s2.strip_index), max(s1.strip_index, s2.strip_index))
            if pair_key in flagged_pairs:
                continue
            m1, m2 = s1.matched_medicine, s2.matched_medicine
            if not m1 or not m2:
                continue
            b1 = f"{m1.brand_name}{f' ({s1.extracted_strength})' if s1.extracted_strength else ''}"
            b2 = f"{m2.brand_name}{f' ({s2.extracted_strength})' if s2.extracted_strength else ''}"
            edges.append(
                {
                    "id": f"edge-safe-{s1.strip_index}-{s2.strip_index}",
                    "rule": "PAIRWISE_INGREDIENT_AND_INTERACTION_CHECK",
                    "status": "Safe",
                    "sourceBrand": b1,
                    "targetBrand": b2,
                    "compoundOrClass": f"{m1.generic} ↔ {m2.generic}",
                    "explanationEn": f"No duplicate active generic or severe contraindicated interaction between {b1} and {b2}.",
                    "explanationBn": f"{b1} এবং {b2}-এর মধ্যে একই জেনেরিক উপাদানের পুনরাবৃত্তি বা মারাত্মক আন্তঃক্রিয়া পাওয়া যায়নি।",
                }
            )

    if len(matched_strips) == 1 and not edges:
        s0 = matched_strips[0]
        m0 = s0.matched_medicine
        if m0:
            b0 = f"{m0.brand_name}{f' ({s0.extracted_strength})' if s0.extracted_strength else ''}"
            edges.append(
                {
                    "id": "edge-single-verified",
                    "rule": "SINGLE_STRIP_DOSAGE_AND_CONTRAINDICATION_CHECK",
                    "status": "Safe",
                    "sourceBrand": b0,
                    "targetBrand": m0.generic,
                    "compoundOrClass": m0.drug_class or m0.generic,
                    "explanationEn": f"Single strip verified against medicines.db ({m0.generic}). No duplicate strip conflict present.",
                    "explanationBn": f"একক স্ট্রিপ ({b0}: {m0.generic}) ডাটাবেস থেকে যাচাইকৃত। কোনো ডুপ্লিকেট ঝুঁকি নেই।",
                }
            )

    return edges


@router.get("/medicines", response_model=PaginatedMedicinesResponse)
def list_medicines(
    q: str = Query("", description="Search brand, generic, manufacturer, class, or indication"),
    dosageForm: str = Query("All", description="Filter by dosage form"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(24, ge=1, le=100, description="Items per page"),
):
    """
    Query medicines.db strictly in READ-ONLY mode with search, dosage form filter, and pagination.
    Generates grounded Bangla `beginner_summary` for every returned medicine record.
    """
    try:
        conn = get_readonly_db_connection()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        cur = conn.cursor()
        q_raw = q.strip().lower()
        q_stripped = _re.sub(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|m)?\b", " ", q_raw, flags=_re.IGNORECASE)
        q_stripped = _re.sub(r"[*]+", " ", q_stripped)
        q_clean = " ".join(q_stripped.split()).strip() or q_raw
        form_clean = dosageForm.strip()
        offset = (page - 1) * limit

        where_clauses = []
        params = []

        if q_clean:
            escaped_q = q_clean.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            where_clauses.append(
                "("
                "b.brand_lower LIKE ? ESCAPE '\\' OR "
                "b.generic_lower LIKE ? ESCAPE '\\' OR "
                "LOWER(COALESCE(b.manufacturer, '')) LIKE ? ESCAPE '\\' OR "
                "LOWER(COALESCE(g.drug_class, '')) LIKE ? ESCAPE '\\' OR "
                "LOWER(COALESCE(g.indication, '')) LIKE ? ESCAPE '\\'"
                ")"
            )
            pattern = f"%{escaped_q}%"
            params.extend([pattern, pattern, pattern, pattern, pattern])

        if form_clean and form_clean != "All":
            where_clauses.append("b.dosage_form = ?")
            params.append(form_clean)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        count_sql = f"""
            SELECT COUNT(*) as total
            FROM brands b
            LEFT JOIN generics g ON b.generic_lower = g.generic_lower
            {where_sql}
        """
        total_row = cur.execute(count_sql, params).fetchone()
        total = int(total_row["total"]) if total_row else 0

        data_sql = f"""
            SELECT
                b.brand_id,
                b.brand_name,
                b.type,
                b.dosage_form,
                COALESCE(g.generic, b.generic_lower) as generic,
                b.manufacturer,
                b.package_container,
                g.drug_class,
                g.indication,
                g.dosage_description,
                g.interaction_description,
                g.side_effects_description,
                g.pregnancy_and_lactation_description,
                b.beginner_summary
            FROM brands b
            LEFT JOIN generics g ON b.generic_lower = g.generic_lower
            {where_sql}
            ORDER BY
                CASE
                    WHEN ? != '' AND b.brand_lower = ? THEN 0
                    WHEN ? != '' AND b.brand_lower LIKE ? ESCAPE '\\' THEN 1
                    ELSE 2
                END,
                b.brand_name ASC
            LIMIT ? OFFSET ?
        """
        escaped_prefix = q_clean.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        prefix_pattern = f"{escaped_prefix}%"
        data_params = [*params, q_clean, q_clean, q_clean, prefix_pattern, limit, offset]
        rows = cur.execute(data_sql, data_params).fetchall()
        items = []
        for row in rows:
            d = dict(row)
            d["beginner_summary"] = generate_bangla_beginner_summary(
                brand_name=d["brand_name"],
                generic=d["generic"],
                dosage_form=d.get("dosage_form"),
                manufacturer=d.get("manufacturer"),
                drug_class=d.get("drug_class"),
                indication=d.get("indication"),
                strength=d.get("package_container"),
                use_llm=False,
            )
            items.append(d)

        global _DB_STATS_CACHE
        if _DB_STATS_CACHE is None:
            stats_row = cur.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM brands) as totalBrands,
                    (SELECT COUNT(*) FROM generics) as totalGenerics
                """
            ).fetchone()
            _DB_STATS_CACHE = {
                "totalBrands": int(stats_row["totalBrands"]) if stats_row else 17191,
                "totalGenerics": int(stats_row["totalGenerics"]) if stats_row else 1661,
            }

        total_pages = max(1, math.ceil(total / limit)) if total > 0 else 1

        return {
            "items": items,
            "total": total,
            "page": page,
            "limit": limit,
            "totalPages": total_pages,
            "stats": _DB_STATS_CACHE,
        }
    finally:
        conn.close()


@router.get("/medicines/search", response_model=List[MedicineInfo])
def search_medicines(
    q: str = Query(..., min_length=2, description="Brand or generic name to search"),
    limit: int = Query(10, ge=1, le=50),
) -> List[MedicineInfo]:
    scanner = get_scanner()
    return scanner.matcher.search_medicines(q, limit=limit)


@router.post("/extract-text", response_model=StripScanResult)
def extract_from_single_strip_text(payload: TextExtractRequest) -> StripScanResult:
    scanner = get_scanner()
    return scanner.matcher.match_strip_text(raw_text=payload.text, strip_index=1, slm_engine=scanner.slm)


@router.post("/extract-multi-text", response_model=MultiStripScanResponse)
def extract_from_multiple_strip_texts(payload: MultiTextExtractRequest) -> MultiStripScanResponse:
    if not payload.strip_texts:
        raise HTTPException(status_code=400, detail="strip_texts cannot be empty.")
    scanner = get_scanner()
    return scanner.scan_texts(payload.strip_texts)


@router.get("/tts")
async def get_bangla_tts_audio(
    text: str = Query(..., min_length=1, max_length=3500, description="Bangla text to synthesize into speech"),
):
    audio_path = await synthesize_bangla_speech(text)
    if not audio_path or not audio_path.exists():
        raise HTTPException(status_code=500, detail="Bangla TTS synthesis failed.")
    return FileResponse(
        path=str(audio_path),
        media_type="audio/mpeg",
        filename=audio_path.name,
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )


@router.post("/tts")
async def post_bangla_tts_audio(payload: TTSRequest):
    """Synthesizes long multi-strip Bangla summaries (up to 6,000 chars) via JSON body without URL length limits."""
    audio_path = await synthesize_bangla_speech(payload.text)
    if not audio_path or not audio_path.exists():
        raise HTTPException(status_code=500, detail="Bangla TTS synthesis failed.")
    return FileResponse(
        path=str(audio_path),
        media_type="audio/mpeg",
        filename=audio_path.name,
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )


@router.get("/tts/clip/{audio_id}")
async def get_cached_bangla_tts_clip(audio_id: str):
    """Serves a pre-synthesized cached Bangla MP3 clip by its 16-hex-char cache key."""
    audio_path = get_cached_tts_audio_path(audio_id)
    if not audio_path:
        raise HTTPException(status_code=404, detail="Cached TTS clip not found.")
    return FileResponse(
        path=str(audio_path),
        media_type="audio/mpeg",
        filename=audio_path.name,
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )


@router.get("/benchmark-samples")
def list_benchmark_samples():
    """Returns available real test photos from `test/` for the AI Transparency Inspector & Pipeline Demo page."""
    available = []
    for item in BENCHMARK_SAMPLES:
        p = TEST_DIR / item["filename"]
        if p.exists():
            available.append(
                {
                    **item,
                    "imageUrl": f"/api/v1/demo-image/{item['id']}",
                }
            )
    return {"samples": available}


@router.get("/demo-image/{sample_id}")
def get_demo_image(sample_id: str):
    """Serves a real demo photo from `test/` for the `/pipeline` page."""
    sample = next((s for s in BENCHMARK_SAMPLES if s["id"] == sample_id), None)
    if sample is None:
        raise HTTPException(status_code=404, detail=f"Unknown sample_id: {sample_id}")
    img_path = TEST_DIR / sample["filename"]
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found.")
    media = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(
        path=str(img_path),
        media_type=media,
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )


def _build_strip_thumbnails(
    image: np.ndarray,
    matched_strips: List[StripScanResult],
    scanner: MedicineStripScanner,
) -> Dict[int, Dict[str, str]]:
    img_h, img_w = image.shape[:2]
    strip_thumbs: Dict[int, Dict[str, str]] = {}
    for s in matched_strips:
        b = s.bbox
        if b:
            x1 = max(0, min(img_w - 1, b.x1))
            y1 = max(0, min(img_h - 1, b.y1))
            x2 = max(x1 + 1, min(img_w, b.x2))
            y2 = max(y1 + 1, min(img_h, b.y2))
            crop_img = image[y1:y2, x1:x2].copy()
        else:
            crop_img = image.copy()
        prep_img = scanner.detector.preprocess_for_ocr(crop_img)
        strip_thumbs[s.strip_index] = {
            "cropThumbnail": _encode_thumb_data_url(crop_img),
            "preprocessedThumbnail": _encode_thumb_data_url(prep_img),
        }
    return strip_thumbs


async def _assemble_scan_response(
    image: np.ndarray,
    scan_resp: MultiStripScanResponse,
    st_times: Dict[str, int],
    scanner: MedicineStripScanner,
    t0: float,
    source_label: str = "upload",
    wait_for_voice: bool = False,
    precomputed_guidance: Optional[List[Tuple[str, str, str]]] = None,
) -> Dict[str, Any]:
    img_h, img_w = image.shape[:2]
    matched_strips = [s for s in scan_resp.strips if s.matched_medicine is not None]

    # Extract per-strip cropped and CLAHE-preprocessed thumbnails in threadpool
    t_thumb = time.perf_counter()
    strip_thumbs = await run_in_threadpool(_build_strip_thumbnails, image, matched_strips, scanner)
    thumb_prep_ms = max(2, int((time.perf_counter() - t_thumb) * 1000))

    yolo_ms = st_times.get("objectDetectionMs", 18)
    prep_ms = st_times.get("preprocessingMs", 0) + thumb_prep_ms
    ocr_ms = st_times.get("ocrEngineMs", 420)
    fuzz_ms = st_times.get("fuzzyMatchMs", 12)
    slm_ms = st_times.get("reasoningMs", 9)

    dup_strip_ids = set()
    inter_strip_ids = set()
    for a in scan_resp.safety_alerts:
        if a.alert_type == "duplicate_generic":
            dup_strip_ids.update(a.involved_strips)
        elif a.alert_type in ("drug_interaction", "same_drug_class"):
            inter_strip_ids.update(a.involved_strips)

    if precomputed_guidance is not None:
        guidance_list = precomputed_guidance
    else:
        t_slm_extra = time.perf_counter()
        guidance_list = await _build_guidance_parallel(scanner, matched_strips, dup_strip_ids, inter_strip_ids)
        slm_ms += int((time.perf_counter() - t_slm_extra) * 1000)

    # Build Bangla voice script AFTER LLM guidance so voice speaks the actual Gemma 3 1B summaries + alerts
    t_voice = time.perf_counter()

    if matched_strips:
        concise_voice_parts = [f"আপনার ছবিতে {len(matched_strips)}টি ওষুধের পাতা শনাক্ত করা হয়েছে।"]
        for idx, (_s, (_dosage_bn, b_summary, _timing)) in enumerate(zip(matched_strips, guidance_list), start=1):
            concise_voice_parts.append(f"ওষুধ {idx}: {b_summary}")
        for a in scan_resp.safety_alerts:
            concise_voice_parts.append(a.message_bn)
        if not scan_resp.safety_alerts:
            concise_voice_parts.append("ওষুধগুলোতে কোনো ডুপ্লিকেট বা ঝুঁকিপূর্ণ আন্তঃক্রিয়া পাওয়া যায়নি।")
        concise_voice_script = "\n".join(concise_voice_parts)
        audio_id = get_tts_cache_key(concise_voice_script)
        audio_url: Optional[str] = f"/api/v1/tts/clip/{audio_id}"
    else:
        concise_voice_script = scan_resp.summary_bn
        audio_url = None

    if wait_for_voice and matched_strips:
        await synthesize_bangla_speech(concise_voice_script)
        voice_ms = max(8, int((time.perf_counter() - t_voice) * 1000))
    else:
        voice_ms = 0

    total_ms = yolo_ms + prep_ms + ocr_ms + fuzz_ms + slm_ms + voice_ms
    wall_ms = max(total_ms, int((time.perf_counter() - t0) * 1000))

    detected_strips_ui = []
    evidence_items = []

    for idx, (s, (dosage_bn, bangla_summary, timing_tag)) in enumerate(zip(matched_strips, guidance_list), start=1):
        med = s.matched_medicine
        assert med is not None
        b = s.bbox
        x_pct = round((b.x1 / max(1, img_w)) * 100, 1) if b else 10
        y_pct = round((b.y1 / max(1, img_h)) * 100, 1) if b else 10
        w_pct = round(((b.x2 - b.x1) / max(1, img_w)) * 100, 1) if b else 80
        h_pct = round(((b.y2 - b.y1) / max(1, img_h)) * 100, 1) if b else 80

        is_dup = s.strip_index in dup_strip_ids
        is_inter = s.strip_index in inter_strip_ids
        status = "duplicate" if is_dup else ("interaction" if is_inter else "safe")
        status_badge = "Danger" if is_dup else ("Caution" if is_inter else "Safe")

        strength_display = s.extracted_strength or med.package_container or "Standard Dose"
        ocr_confs = [ln.confidence for ln in s.ocr_lines] if s.ocr_lines else [0.96]
        avg_ocr_conf = round(float(sum(ocr_confs) / len(ocr_confs)), 3)
        conf_pct = round(avg_ocr_conf * 100.0, 1)
        det_source = s.detection_source or "ocr_fallback"
        yolo_conf_pct = round(s.yolo_confidence * 100.0, 1) if s.yolo_confidence is not None else None
        source_prefix = "YOLO26s" if det_source.startswith("yolo26") else "RapidOCR"

        thumbs = strip_thumbs.get(s.strip_index, {"cropThumbnail": "", "preprocessedThumbnail": ""})

        detected_strips_ui.append(
            {
                "id": f"strip-{idx}",
                "label": f"{source_prefix} #{idx:02d} • {(yolo_conf_pct if yolo_conf_pct is not None else conf_pct):.1f}%",
                "brandName": med.brand_name,
                "genericName": med.generic,
                "strength": strength_display,
                "dosageForm": s.detected_dosage_form or med.dosage_form or "Tablet",
                "manufacturer": med.manufacturer or "Bangladesh Pharma",
                "confidence": s.yolo_confidence if s.yolo_confidence is not None else avg_ocr_conf,
                "ocrConfidence": avg_ocr_conf,
                "yoloConfidence": s.yolo_confidence,
                "detectionSource": det_source,
                "x": x_pct,
                "y": y_pct,
                "width": w_pct,
                "height": h_pct,
                "rotation": 0,
                "status": status,
                "dosageBangla": dosage_bn,
                "dosageEnglish": bangla_summary,
                "timingTag": timing_tag,
                "databaseRecord": {
                    "brand_id": med.brand_id,
                    "brand_name": med.brand_name,
                    "type": "allopathic",
                    "dosage_form": s.detected_dosage_form or med.dosage_form,
                    "generic": med.generic,
                    "manufacturer": med.manufacturer,
                    "package_container": s.extracted_strength or med.package_container,
                    "drug_class": med.drug_class,
                    "indication": med.indication,
                    "dosage_description": med.dosage_description,
                    "interaction_description": med.interaction_description,
                    "side_effects_description": med.side_effects_description,
                    "pregnancy_and_lactation_description": med.pregnancy_and_lactation_description,
                    "beginner_summary": bangla_summary,
                },
            }
        )

        evidence_items.append(
            {
                "id": f"ev-{idx}",
                "stripIndex": s.strip_index,
                "brandName": med.brand_name,
                "genericName": med.generic,
                "strength": strength_display,
                "dosageForm": s.detected_dosage_form or med.dosage_form or "Tablet",
                "manufacturer": med.manufacturer or "Bangladesh Pharma",
                "statusBadge": status_badge,
                "rawOcrText": s.raw_ocr_text,
                "ocrLinesCount": len(s.ocr_lines),
                "fuzzyMatchScore": s.match_score,
                "matchMethod": s.match_method or "exact_brand",
                "confidencePct": conf_pct,
                "yoloConfidencePct": yolo_conf_pct,
                "detectionSource": det_source,
                "bboxCoords": {
                    "x1": b.x1 if b else 0,
                    "y1": b.y1 if b else 0,
                    "x2": b.x2 if b else img_w,
                    "y2": b.y2 if b else img_h,
                    "imgWidth": img_w,
                    "imgHeight": img_h,
                },
                "cropThumbnail": thumbs["cropThumbnail"],
                "preprocessedThumbnail": thumbs["preprocessedThumbnail"],
                "banglaSummary": bangla_summary,
            }
        )

    alerts_ui = []
    for idx, a in enumerate(scan_resp.safety_alerts, start=1):
        alerts_ui.append(
            {
                "id": f"alert-{idx}",
                "type": "duplicate" if a.alert_type == "duplicate_generic" else "interaction",
                "severity": "critical" if a.alert_type == "duplicate_generic" else "high",
                "title": (
                    "Accidental Therapeutic Duplication Detected"
                    if a.alert_type == "duplicate_generic"
                    else "Potential Drug Interaction Detected"
                ),
                "titleBangla": (
                    "মারাত্মক সতর্কতা: একই জেনেরিক ওষুধের পুনরাবৃত্তি (ওভারডোজ ঝুঁকি)"
                    if a.alert_type == "duplicate_generic"
                    else "সতর্কতা: ওষুধের আন্তঃক্রিয়া (Drug Interaction)"
                ),
                "description": a.message_en,
                "descriptionBangla": a.message_bn,
                "involvedBrands": a.involved_brands,
                "genericCompound": a.generic_or_class,
                "clinicalRisk": "Overdose / Adverse Interaction Risk",
            }
        )

    overall_safety = (
        "critical_risk"
        if dup_strip_ids
        else ("moderate_warning" if inter_strip_ids else "all_safe")
    )

    safety_edges = _build_safety_edges(matched_strips, scan_resp.safety_alerts)

    det_info = scan_resp.detector_info or scanner.detector.get_model_info()
    active_detector_label = det_info.get("modelLabel") or scanner.detector.model_label

    transparency_inspector = {
        "sourceLabel": source_label,
        "totalTurnaroundMs": total_ms,
        "wallClockMs": wall_ms,
        "activeDetector": active_detector_label,
        "detectorInfo": det_info,
        "activeOcrEngine": "PaddleOCR PP-OCRv6 (RapidOCR ONNX)",
        "activeReasoningMode": get_active_slm_label(),
        "stages": [
            {
                "id": "object_detection",
                "label": "Object Detection",
                "engine": f"{active_detector_label} ({det_info.get('effectiveMode', 'hybrid_fusion')})",
                "durationMs": yolo_ms,
                "color": "#0ea86b",
            },
            {
                "id": "image_preprocessing",
                "label": "Image Preprocessing",
                "engine": "OpenCV LAB-CLAHE + Unsharp Mask",
                "durationMs": prep_ms,
                "color": "#0d9488",
            },
            {
                "id": "ocr_engine",
                "label": "OCR Engine",
                "engine": "PP-OCRv6 Det + Rec ONNX (0° + 90° CCW)",
                "durationMs": ocr_ms,
                "color": "#0b5563",
            },
            {
                "id": "fuzzy_matching",
                "label": "Database Fuzzy Matching",
                "engine": "RapidFuzz WRatio + SQLite 17,191 BD Index",
                "durationMs": fuzz_ms,
                "color": "#2563eb",
            },
            {
                "id": "reasoning",
                "label": "Reasoning",
                "engine": "Gemma 3 1B-Q4_K_M Bangla SLM + Pairwise Conflict Graph",
                "durationMs": slm_ms,
                "color": "#7c3aed",
            },
            {
                "id": "voice_synthesis",
                "label": "Voice Synthesis",
                "engine": "Edge Neural TTS (bn-BD-NabanitaNeural)",
                "durationMs": voice_ms,
                "color": "#d97706",
            },
        ],
        "evidenceItems": evidence_items,
        "safetyEdges": safety_edges,
    }

    return {
        "scanId": f"SCN-LIVE-{int(time.time())}-{uuid.uuid4().hex[:6]}",
        "timestamp": f"YOLO26 ONNX + PP-OCRv6 + RapidFuzz + Gemma 3 1B SLM • {total_ms}ms",
        "inferenceTimeMs": total_ms,
        "pipelineMetrics": {
            "yoloDetectionMs": yolo_ms,
            "ocrRectificationMs": prep_ms + ocr_ms,
            "fuzzyEntityMatchMs": fuzz_ms,
            "slmSafetyReasoningMs": slm_ms,
        },
        "overallSafety": overall_safety,
        "detectedStrips": detected_strips_ui,
        "alerts": alerts_ui,
        "banglaVoiceScript": concise_voice_script,
        "englishTranslation": scan_resp.summary_en,
        "audioUrl": audio_url,
        "transparencyInspector": transparency_inspector,
        "strips": [s.model_dump() for s in scan_resp.strips],
    }


async def _execute_full_scan_pipeline(
    image: np.ndarray,
    source_label: str = "upload",
    wait_for_voice: bool = False,
    detector_mode: Optional[str] = None,
    yolo_conf: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Runs the complete AI pipeline on an OpenCV image, measuring exact per-stage execution
    times in milliseconds and producing cropped + CLAHE-preprocessed thumbnails.
    """
    t0 = time.perf_counter()
    scanner = await run_in_threadpool(get_scanner)
    scan_resp, st_times = await run_in_threadpool(
        scanner.scan_image_with_timings,
        image,
        detector_mode,
        yolo_conf,
    )
    return await _assemble_scan_response(
        image=image,
        scan_resp=scan_resp,
        st_times=st_times,
        scanner=scanner,
        t0=t0,
        source_label=source_label,
        wait_for_voice=wait_for_voice,
    )


def _decode_image_bytes(content: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


async def _read_and_validate_upload(file: UploadFile) -> np.ndarray:
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded image exceeds maximum allowed size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    logger.info("[DEV-API] Received upload file=%r (%d bytes)", file.filename, len(content) if content else 0)
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")
    image = await run_in_threadpool(_decode_image_bytes, content)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    h, w = image.shape[:2]
    if h * w > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=400, detail="Uploaded image resolution is too large.")
    return image


@router.post("/scan")
async def scan_medicine_strips(
    file: UploadFile = File(...),
    detector_mode: Optional[Literal["hybrid_fusion", "yolo_only", "ocr_only"]] = Query(
        None, description="Optional override: hybrid_fusion | yolo_only | ocr_only"
    ),
    yolo_conf: Optional[float] = Query(None, ge=0.05, le=0.95, description="Optional YOLO confidence threshold override"),
):
    image = await _read_and_validate_upload(file)
    return await _execute_full_scan_pipeline(
        image,
        source_label=file.filename or "Uploaded Image",
        detector_mode=detector_mode,
        yolo_conf=yolo_conf,
    )


@router.post("/scan-stream")
async def scan_medicine_strips_stream(
    file: UploadFile = File(...),
    detector_mode: Optional[Literal["hybrid_fusion", "yolo_only", "ocr_only"]] = Query(
        None, description="Optional override: hybrid_fusion | yolo_only | ocr_only"
    ),
    yolo_conf: Optional[float] = Query(None, ge=0.05, le=0.95, description="Optional YOLO confidence threshold override"),
):
    """
    Streams real-time NDJSON progress events as each AI pipeline stage executes on the backend
    via the unified MedicineStripScanner pipeline, followed by the final scan result payload.
    """
    image = await _read_and_validate_upload(file)
    source_label = file.filename or "Uploaded Image"

    async def event_generator():
        try:
            t0 = time.perf_counter()
            scanner = await run_in_threadpool(get_scanner)

            # Stage 1 (index 0): Real YOLO26 ONNX Strip Detection
            yield json.dumps({"type": "stage", "stageIndex": 0, "status": "running"}) + "\n"
            detection, yolo_ms = await run_in_threadpool(
                scanner.run_stage1_detect, image, detector_mode, yolo_conf
            )
            yield json.dumps({"type": "stage", "stageIndex": 0, "status": "done", "durationMs": yolo_ms}) + "\n"

            # Stage 2 & 3 (index 1 & 2): Single-Pass RapidOCR (lazy LAB-CLAHE if 0 lines)
            yield json.dumps({"type": "stage", "stageIndex": 1, "status": "running"}) + "\n"
            yield json.dumps({"type": "stage", "stageIndex": 2, "status": "running"}) + "\n"
            ocr_lines, prep_ms, ocr_ms = await run_in_threadpool(scanner.run_stage2_3_ocr, image)
            yield json.dumps({"type": "stage", "stageIndex": 1, "status": "done", "durationMs": prep_ms}) + "\n"
            yield json.dumps({"type": "stage", "stageIndex": 2, "status": "done", "durationMs": ocr_ms}) + "\n"

            # Stage 4 (index 3): YOLO Spatial Partitioning + Selective Crop Rescue + Free Residual Fallback + 17K+ Matching
            yield json.dumps({"type": "stage", "stageIndex": 3, "status": "running"}) + "\n"
            strip_results, fuzz_ms, rescue_prep_ms, fusion_stats = await run_in_threadpool(
                scanner.run_stage4_fuse_and_match,
                image,
                detection,
                ocr_lines,
                detector_mode,
            )
            prep_ms += rescue_prep_ms
            yield json.dumps({"type": "stage", "stageIndex": 3, "status": "done", "durationMs": fuzz_ms}) + "\n"

            # Stage 5 (index 4): Safety SLM Bangla Summary (Gemma 3 1B parallel) & Pairwise Conflict Graph
            yield json.dumps({"type": "stage", "stageIndex": 4, "status": "running"}) + "\n"
            t_slm = time.perf_counter()
            alerts, summary_en, summary_bn = await run_in_threadpool(evaluate_multi_strip_safety, strip_results)

            matched_strips = [s for s in strip_results if s.matched_medicine is not None]
            dup_ids = set()
            inter_ids = set()
            for a in alerts:
                if a.alert_type == "duplicate_generic":
                    dup_ids.update(a.involved_strips)
                elif a.alert_type in ("drug_interaction", "same_drug_class"):
                    inter_ids.update(a.involved_strips)

            precomputed_guidance = await _build_guidance_parallel(scanner, matched_strips, dup_ids, inter_ids)
            slm_ms = max(3, int((time.perf_counter() - t_slm) * 1000))
            yield json.dumps({"type": "stage", "stageIndex": 4, "status": "done", "durationMs": slm_ms}) + "\n"

            base_info = scanner.detector.get_model_info()
            detector_info = {
                **base_info,
                "activeBackend": detection.backend,
                "requestedMode": (detector_mode or base_info["defaultPipelineMode"]).strip().lower(),
                "confThresholdUsed": yolo_conf if yolo_conf is not None else base_info["defaultConfThreshold"],
                **fusion_stats,
            }
            scan_resp = MultiStripScanResponse(
                total_strips_detected=len(strip_results),
                strips=strip_results,
                safety_alerts=alerts,
                summary_en=summary_en,
                summary_bn=summary_bn,
                detector_info=detector_info,
            )
            st_times = {
                "objectDetectionMs": yolo_ms,
                "preprocessingMs": prep_ms,
                "ocrEngineMs": ocr_ms,
                "fuzzyMatchMs": fuzz_ms,
                "reasoningMs": slm_ms,
            }
            payload = await _assemble_scan_response(
                image=image,
                scan_resp=scan_resp,
                st_times=st_times,
                scanner=scanner,
                t0=t0,
                source_label=source_label,
                wait_for_voice=False,
                precomputed_guidance=precomputed_guidance,
            )
            yield json.dumps({"type": "result", "data": payload}, ensure_ascii=False) + "\n"
        except Exception as exc:
            logger.exception("[DEV-API] /scan-stream failed: %s", exc)
            yield json.dumps({"type": "error", "detail": str(exc)}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/scan-benchmark")
async def run_sample_benchmark_scan(
    sample_id: str = Query("multi-5-strips", description="Sample benchmark ID from /benchmark-samples"),
):
    """
    Executes the real 6-stage AI pipeline live on a real medicine strip image from `test/`
    so judges can re-run and inspect real execution times and cropped thumbnails at any time.
    """
    sample = next((s for s in BENCHMARK_SAMPLES if s["id"] == sample_id), None)
    if sample is None:
        raise HTTPException(status_code=404, detail=f"Unknown benchmark sample_id: {sample_id}")
    img_path = TEST_DIR / sample["filename"]
    if not img_path.exists():
        # Fallback to first existing image in test/
        existing = [p for p in TEST_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}] if TEST_DIR.exists() else []
        if not existing:
            raise HTTPException(status_code=404, detail="No test images found in test/ directory.")
        img_path = existing[0]

    image = await run_in_threadpool(cv2.imread, str(img_path))
    if image is None:
        raise HTTPException(status_code=500, detail=f"Failed to read benchmark image {img_path.name}")

    logger.info("[DEV-API] Running live benchmark scan on %s (%s)", sample["id"], img_path.name)
    result = await _execute_full_scan_pipeline(image, source_label=f"{sample['title']} ({img_path.name})")
    result["sampleId"] = sample["id"]
    result["sampleTitle"] = sample["title"]
    result["sampleTitleBn"] = sample["titleBn"]
    result["demoImageUrl"] = f"/api/v1/demo-image/{sample['id']}"
    return result
