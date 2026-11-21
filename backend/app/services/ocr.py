from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
import logging
import math
import re
import sqlite3
import cv2
import numpy as np

try:
    from backend.app.core.config import DB_PATH, MIN_FUZZY_SCORE, OCR_LANG
    from backend.app.models.schemas import (
        BoundingBox,
        MedicineInfo,
        MultiStripScanResponse,
        OCRLine,
        SafetyAlert,
        StripScanResult,
    )
    from backend.app.services.detector import StripDetector
except ImportError:
    from app.core.config import DB_PATH, MIN_FUZZY_SCORE, OCR_LANG  # type: ignore
    from app.models.schemas import (  # type: ignore
        BoundingBox,
        MedicineInfo,
        MultiStripScanResponse,
        OCRLine,
        SafetyAlert,
        StripScanResult,
    )
    from app.services.detector import StripDetector  # type: ignore

from rapidfuzz import fuzz, process

logger = logging.getLogger("oushudhcheck.scanner")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ---------------------------------------------------------------------------
# 1. Regulatory Noise Removal & Strength / Dosage Extractor
# ---------------------------------------------------------------------------

# Strip regulatory license / MA / DAR / batch strings so their digits ('Mfg. Lic. 66 & 247', 'MA No. 052-79-046')
# are never mistaken for drug strengths.
_REGULATORY_LINE_RE = re.compile(
    r"(?:"
    r"mfg\.?\s*lic\.?\s*(?:no\.?)?\s*[:.]?\s*[\w\-\s&:]+"
    r"|m\.?a\.?\s*no\.?\s*[:.]?\s*[\d\-\s]+"
    r"|d\.?a\.?r\.?\s*no\.?\s*[:.]?\s*[\d\-\s]+"
    r"|\b\d{2,3}\s*&\s*\d{2,3}\b"
    r"|\b\d{3}-\d{2,4}-\d{2,4}\b"
    r"|\b\d[0-9A-Za-z]{6,14}\b"
    r")",
    re.IGNORECASE,
)

# Common foil OCR misreads near units ('5OO rng', 'Emistat8 my', '500 mq')
_OCR_DIGIT_O_RE = re.compile(
    r"(?<=\d)[Oo]+(?=\d|\s*(?:mg|rng|mq|my|g|mcg|rncg|ug|µg|ml|rnl|iu|%|w/v|w/w)\b)"
)
_OCR_UNIT_FIX_RE = [
    (re.compile(r"(?<=\d)\s*(?:rng|mq|my|m9)\b", re.IGNORECASE), " mg"),
    (re.compile(r"(?<=\d)\s*rncg\b", re.IGNORECASE), " mcg"),
    (re.compile(r"(?<=\d)\s*rnl\b", re.IGNORECASE), " ml"),
]

_UNIT_PAT = r"(?:mg|mcg|ug|µg|g|kg|ml|l|iu|u|%\s*(?:w/v|w/w|v/v)?)"

# Combination strength pattern: e.g. "500 mg + 65 mg", "500mg/65mg", "200 mg / 5 ml"
_COMBO_STRENGTH_RE = re.compile(
    rf"(?<![\d.])(\d+(?:[.,]\d+)?)\s*({_UNIT_PAT})?\s*(?:\+|/)\s*(\d+(?:[.,]\d+)?)\s*({_UNIT_PAT})\b",
    re.IGNORECASE,
)

# Single strength pattern: note `(?<![\d.])` allows matching when attached to brand letters like "Emistat8 mg" or "50mg"
_SINGLE_STRENGTH_RE = re.compile(
    rf"(?<![\d.])(\d+(?:[.,]\d+)?)\s*({_UNIT_PAT})\b",
    re.IGNORECASE,
)

# Common Bangladeshi medicine strength numbers when printed right after brand name without 'mg'
# e.g. "Naprosyn®500", "Prelica®50", "SeroneX50", "Fluver 10", "Epitra°1"
_KNOWN_BARE_MG_VALUES = {
    "0.25", "0.5", "0.75", "1", "1.5", "2", "2.5", "3", "4", "5", "6", "8",
    "10", "12.5", "15", "20", "25", "30", "40", "50", "60", "65", "75", "80",
    "90", "100", "120", "125", "150", "200", "250", "300", "375", "400",
    "500", "600", "625", "650", "665", "750", "850", "1000",
}

_DOSAGE_FORM_KEYWORDS: Dict[str, str] = {
    "tablet": "Tablet",
    "tablets": "Tablet",
    "tab": "Tablet",
    "tabs": "Tablet",
    "capsule": "Capsule",
    "capsules": "Capsule",
    "cap": "Capsule",
    "caps": "Capsule",
    "syrup": "Syrup",
    "suspension": "Suspension",
    "injection": "Injection",
    "inj": "Injection",
    "gel": "Gel",
    "cream": "Cream",
    "ointment": "Ointment",
    "drops": "Drop",
    "drop": "Drop",
    "inhaler": "Inhaler",
    "suppository": "Suppository",
    "sachet": "Sachet",
}

_STRIP_STOPWORDS = {
    "bp", "usp", "ip", "inn", "ph", "eur", "jp", "bpc", "type", "ii",
    "mfg", "mig", "lic", "no", "dar", "ma", "batch", "lot", "exp", "mfd", "mrp", "tk", "bdt", "price",
    "each", "film", "coated", "enteric", "sustained", "extended", "release", "chewable",
    "dispersible", "oro", "mouth", "dissolving", "contains", "equivalent", "to", "as",
    "tablet", "tablets", "tab", "tabs", "capsule", "capsules", "cap", "caps", "syrup",
    "suspension", "injection", "gel", "cream", "ointment", "drops", "blister", "strip",
    "keep", "out", "of", "reach", "children", "store", "below", "cool", "dry", "place",
    "away", "from", "light", "heat", "moisture", "protect", "dosage", "directed", "by",
    "physician", "doctor", "registered", "trademark", "manufactured", "marketed",
    "distributed", "for", "only", "oral", "use", "hydrochloride", "hydrochioride", "hydrocharide",
    "sodium", "sulfate", "sulfale", "sufale", "magnesium", "trihydrate", "sesquihydrate",
    "ltd", "limited", "lumited", "pharmaceuticals", "pharmaceutical", "phammaceuticals", "pharma",
    "laboratories", "laboratory", "labs", "healthcare", "althcare", "ealthcare", "euticals",
    "health", "care", "drugs", "drug", "international", "bangladesh", "dhaka", "tongi", "longi",
    "gazipur", "gazir", "bscic", "rajendrapur",
    "beximco", "square", "incepta", "renata", "acme", "eskayef", "skf", "aristopharma",
    "opsonin", "aci", "popular", "radiant", "radant", "radient", "raoant", "ziska", "general",
    "beacon", "ibn", "sina", "unimed", "unihealth", "orion", "nipro", "jmi", "sanofi",
    "novartis", "gsk", "novo", "nordisk", "roche", "pfizer", "astrazeneca", "servier",
    "sun", "biopharma", "navana", "globe", "delta", "somatec", "sharif", "kemiko", "alco",
    "pacific", "apex", "silva", "white", "horse", "labaid", "bpl", "spc", "plc",
}


def normalize_ocr_for_strength(text: str) -> str:
    """Cleans regulatory numbers and fixes foil OCR typos near dosage units."""
    if not text:
        return ""
    # Replace newlines with spaces so split tokens like 'Emistat8' + '\n' + 'mg' become 'Emistat8 mg'
    cleaned = " ".join(text.splitlines())
    cleaned = _REGULATORY_LINE_RE.sub(" ", cleaned)

    for _ in range(4):
        new_cleaned = _OCR_DIGIT_O_RE.sub(lambda m: "0" * len(m.group(0)), cleaned)
        if new_cleaned == cleaned:
            break
        cleaned = new_cleaned

    for pat, repl in _OCR_UNIT_FIX_RE:
        cleaned = pat.sub(repl, cleaned)
    return cleaned


def _format_unit(unit: str, val_str: str = "", has_mg_context: bool = False) -> str:
    u = unit.strip().lower()
    if u == "iu":
        return "IU"
    if u in ("ug", "µg"):
        return "mcg"
    # On a tablet/capsule blister strip, if OCR drops 'm' from '20 mg' -> '20 g', fix 'g' -> 'mg'
    if u == "g":
        try:
            v = float(val_str)
            if v >= 5.0 or has_mg_context:
                return "mg"
        except ValueError:
            pass
    return u


def extract_strengths(text: str, brand_name: Optional[str] = None) -> Tuple[Optional[str], List[str]]:
    """
    Extracts medicine strength(s) (e.g. '500 mg', '1000 mg', '50 mg', '10 mg', '8 mg', '1 mg',
    '20 mg + 750 mg + 50 mg + 0.75 mg') from strip OCR text.
    """
    if not text or not text.strip():
        return None, []

    norm_text = normalize_ocr_for_strength(text)
    has_mg_context = bool(re.search(r"\bmg\b", norm_text, re.IGNORECASE))
    found: List[str] = []
    spans_to_mask: List[Tuple[int, int]] = []

    # 1. Explicit combination strengths (e.g. "500 mg + 65 mg", "200 mg / 5 ml")
    for m in _COMBO_STRENGTH_RE.finditer(norm_text):
        val1 = m.group(1).replace(",", ".")
        unit2 = _format_unit(m.group(4), m.group(3), has_mg_context)
        unit1 = _format_unit(m.group(2), val1, has_mg_context) if m.group(2) else unit2
        val2 = m.group(3).replace(",", ".")
        sep = " + " if "+" in m.group(0) else " / "
        formatted = f"{val1} {unit1}{sep}{val2} {unit2}"
        if formatted not in found:
            found.append(formatted)
        spans_to_mask.append(m.span())

    masked = list(norm_text)
    for start, end in spans_to_mask:
        for i in range(start, end):
            masked[i] = " "
    remaining_text = "".join(masked)

    # 2. Explicit single strengths with unit (e.g. "500 mg", "1000 mg", "8 mg", "1 mg", "750 mg")
    for m in _SINGLE_STRENGTH_RE.finditer(remaining_text):
        val = m.group(1).replace(",", ".")
        try:
            if float(val) <= 0:
                continue
        except ValueError:
            continue
        unit = _format_unit(m.group(2), val, has_mg_context)
        formatted = f"{val} {unit}"
        if formatted not in found:
            found.append(formatted)

    # 3. Bare strength number attached to or right after brand name / trademark symbol
    # e.g., "Prelica®50", "SeroneX50", "Fluver\"10", "Naprosyn®500", "Epitra°1"
    if not found:
        if brand_name:
            b_pat = re.escape(brand_name.split()[0])
            for m in re.finditer(rf"{b_pat}[A-Za-z]*[\s®°\"']*(\d+(?:\.\d+)?)\b", norm_text, re.IGNORECASE):
                tok = m.group(1)
                if tok in _KNOWN_BARE_MG_VALUES:
                    candidate = f"{tok} mg"
                    if candidate not in found:
                        found.append(candidate)

        # Also check any word/symbol followed by a known bare mg number (e.g. "®50", "X50", "\"10")
        if not found:
            for m in re.finditer(r"(?:[A-Za-z]{3,}[\s®°\"']*|[®°])(\d+(?:\.\d+)?)\b", norm_text):
                tok = m.group(1)
                if tok in _KNOWN_BARE_MG_VALUES:
                    candidate = f"{tok} mg"
                    if candidate not in found:
                        found.append(candidate)

        if not found:
            for tok in re.findall(r"\b(\d+(?:\.\d+)?)\b", norm_text):
                if tok in _KNOWN_BARE_MG_VALUES:
                    candidate = f"{tok} mg"
                    if candidate not in found:
                        found.append(candidate)

    primary = found[0] if found else None
    # Join multi-ingredient strengths if separated by '+', '&', 'and', or ',' (e.g. Orthogen's 4 active ingredients)
    if len(found) > 1 and ("+" in norm_text or "&" in norm_text or " and " in norm_text.lower()):
        primary = " + ".join(found[:4])

    return primary, found


def detect_dosage_form(text: str) -> Optional[str]:
    if not text:
        return None
    words = re.findall(r"[a-zA-Z]+", text.lower())
    for w in words:
        if w in _DOSAGE_FORM_KEYWORDS:
            return _DOSAGE_FORM_KEYWORDS[w]
    return None


def strip_strengths_and_noise(text: str) -> str:
    """
    Removes regulatory codes, strength tokens, digits (even when attached to letters like 'SeroneX50',
    'Fluver10', 'Emistat8', 'Prelica®50'), and pharma stopwords to isolate brand/generic words.
    """
    norm = normalize_ocr_for_strength(text)
    norm = _COMBO_STRENGTH_RE.sub(" ", norm)
    norm = _SINGLE_STRENGTH_RE.sub(" ", norm)
    # Extract purely alphabetic words (splitting letters from attached numbers like 'SeroneX50' -> 'SeroneX')
    tokens = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", norm)
    filtered = [
        t
        for t in tokens
        if t.lower() not in _STRIP_STOPWORDS and len(t) >= 3
    ]
    return " ".join(filtered).strip()


# ---------------------------------------------------------------------------
# 2. RapidOCR Engine (PP-OCRv6 ONNX Primary + Multi-Orientation Support)
# ---------------------------------------------------------------------------


class OCREngine:
    """
    Primary OCR engine powered by RapidOCR (PP-OCRv6 detection + recognition ONNX models).
    Extracts text lines with exact bounding-box coordinates and supports multi-orientation
    scanning (0° + 90° CCW) for vertically printed blister strips.
    """

    def __init__(self, lang: str = OCR_LANG) -> None:
        self.lang = lang
        self._backend_name: Optional[str] = None
        self._engine = None
        self._init_engine()

    def _init_engine(self) -> None:
        # 1. RapidOCR (PP-OCRv6 ONNX — fast, accurate on reflective medicine strips)
        try:
            try:
                from rapidocr import RapidOCR  # type: ignore
            except ImportError:
                from rapidocr_onnxruntime import RapidOCR  # type: ignore

            self._engine = RapidOCR()
            self._backend_name = "rapidocr"
            logger.info("[DEV-OCR] Initialized RapidOCR (PP-OCRv6 ONNXRuntime) engine")
            return
        except Exception as exc:
            logger.warning("[DEV-OCR] RapidOCR init failed: %s", exc)

        # 2. PaddleOCR fallback
        try:
            from paddleocr import PaddleOCR  # type: ignore

            self._engine = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
            self._backend_name = "paddleocr"
            logger.info("[DEV-OCR] Initialized PaddleOCR engine")
            return
        except Exception:
            pass

        # 3. EasyOCR fallback
        try:
            import easyocr  # type: ignore

            self._engine = easyocr.Reader([self.lang], gpu=False, verbose=False)
            self._backend_name = "easyocr"
            logger.info("[DEV-OCR] Initialized EasyOCR engine")
            return
        except Exception:
            self._backend_name = None

    def _run_single_pass(self, image: np.ndarray) -> List[Tuple[List[List[float]], str, float]]:
        """Returns list of `(4_point_box, text, confidence)` from a single image pass."""
        if image is None or image.size == 0 or self._backend_name is None:
            return []

        raw_items: List[Tuple[List[List[float]], str, float]] = []

        if self._backend_name == "rapidocr":
            out = self._engine(image)
            if hasattr(out, "txts") and out.txts:
                boxes = getattr(out, "boxes", None)
                scores = getattr(out, "scores", None) or [0.9] * len(out.txts)
                for idx, txt in enumerate(out.txts):
                    txt_str = str(txt).strip()
                    if not txt_str:
                        continue
                    conf = float(scores[idx]) if idx < len(scores) else 0.9
                    box = boxes[idx].tolist() if (boxes is not None and idx < len(boxes)) else [[0, 0], [100, 0], [100, 20], [0, 20]]
                    raw_items.append((box, txt_str, conf))
            elif isinstance(out, tuple) and out[0]:
                for item in out[0]:
                    box, txt_str, conf = item[0], str(item[1]).strip(), float(item[2])
                    if txt_str:
                        raw_items.append((box, txt_str, conf))

        elif self._backend_name == "paddleocr":
            result = self._engine.ocr(image, cls=True)
            if result and result[0]:
                for item in result[0]:
                    box, (txt_str, conf) = item[0], item[1]
                    txt_str = str(txt_str).strip()
                    if txt_str:
                        raw_items.append((box, txt_str, float(conf)))

        elif self._backend_name == "easyocr":
            result = self._engine.readtext(image)
            for box, txt_str, conf in result:
                txt_str = str(txt_str).strip()
                if txt_str:
                    raw_items.append((box, txt_str, float(conf)))

        return raw_items

    def extract_lines(self, image: np.ndarray, include_rotated: bool = True) -> List[OCRLine]:
        """
        Extracts OCR text lines with original-image bounding boxes.
        Also runs a 90° counter-clockwise pass (mapping boxes back to original coordinates)
        so vertically oriented blister strips (e.g. Naprosyn 500, Fluver 10) are captured with high accuracy.
        """
        if image is None or image.size == 0 or self._backend_name is None:
            return []

        h, w = image.shape[:2]
        lines: List[OCRLine] = []

        # Pass 1: 0° original orientation
        for box, txt, conf in self._run_single_pass(image):
            if conf < 0.50:
                continue
            xs = [float(pt[0]) for pt in box]
            ys = [float(pt[1]) for pt in box]
            x1, y1, x2, y2 = int(max(0, min(xs))), int(max(0, min(ys))), int(min(w, max(xs))), int(min(h, max(ys)))
            box_h = float(max(y2 - y1, x2 - x1))
            lines.append(
                OCRLine(
                    text=txt,
                    confidence=round(conf, 3),
                    box_height=round(box_h, 1),
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=round(conf, 3)),
                )
            )

        # Pass 2: 90° counter-clockwise pass to pick up vertical text lines (mapped back to original image coords)
        if include_rotated:
            rot_ccw = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            existing_texts_lower = {ln.text.lower() for ln in lines}
            for box, txt, conf in self._run_single_pass(rot_ccw):
                if conf < 0.72:
                    continue
                # Map (rx, ry) in 90-CCW rotated image (size W x H) back to (orig_x, orig_y) in (H x W):
                # In ROTATE_90_COUNTERCLOCKWISE: rx = orig_y, ry = W - 1 - orig_x -> orig_x = W - 1 - ry, orig_y = rx
                orig_xs = [float(w - 1 - pt[1]) for pt in box]
                orig_ys = [float(pt[0]) for pt in box]
                x1 = int(max(0, min(orig_xs)))
                y1 = int(max(0, min(orig_ys)))
                x2 = int(min(w, max(orig_xs)))
                y2 = int(min(h, max(orig_ys)))
                if txt.lower() not in existing_texts_lower:
                    box_h = float(max(y2 - y1, x2 - x1))
                    lines.append(
                        OCRLine(
                            text=txt,
                            confidence=round(conf, 3),
                            box_height=round(box_h, 1),
                            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=round(conf, 3)),
                        )
                    )

        logger.info(
            "[DEV-OCR] Extracted %d OCR lines: %s",
            len(lines),
            [(ln.text, ln.confidence) for ln in lines],
        )
        return lines


# ---------------------------------------------------------------------------
# 3. Medicine Database Matcher (RapidFuzz + Read-Only SQLite medicines.db)
# ---------------------------------------------------------------------------


@dataclass
class _BrandIndexEntry:
    brand_id: int
    brand_name: str
    brand_lower: str
    generic: str
    generic_lower: str
    dosage_form: Optional[str]
    manufacturer: Optional[str]
    package_container: Optional[str]


class MedicineMatcher:
    """
    Matches noisy OCR text from medicine blister strips against the updated
    `data/processed/medicines.db` (17,191 distinct medicines + `beginner_summary`).
    Opens `medicines.db` strictly in read-only mode.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._brands_by_lower: Dict[str, List[_BrandIndexEntry]] = {}
        self._brands_by_id: Dict[int, _BrandIndexEntry] = {}
        self._unique_brand_lowers: List[str] = []
        self._unique_generic_lowers: List[str] = []
        self._generic_first_words: Set[str] = set()
        self._load_index()

    def _load_index(self) -> None:
        if not self.db_path.exists():
            logger.warning("[DEV-DB] Database not found at %s", self.db_path)
            return

        with sqlite3.connect(f"{self.db_path.as_uri()}?mode=ro", uri=True) as conn:
            cur = conn.cursor()
            rows = cur.execute(
                """
                SELECT brand_id, brand_name, brand_lower, generic, generic_lower,
                       dosage_form, manufacturer, package_container
                FROM medicines
                """
            ).fetchall()

        generics_set: Set[str] = set()
        for r in rows:
            entry = _BrandIndexEntry(
                brand_id=int(r[0]),
                brand_name=r[1],
                brand_lower=r[2],
                generic=r[3],
                generic_lower=r[4],
                dosage_form=r[5],
                manufacturer=r[6],
                package_container=r[7],
            )
            self._brands_by_id[entry.brand_id] = entry
            self._brands_by_lower.setdefault(entry.brand_lower, []).append(entry)
            generics_set.add(entry.generic_lower)
            for part in re.split(r"[\s+/&,]+", entry.generic_lower):
                if len(part) >= 5:
                    self._generic_first_words.add(part)

        self._unique_brand_lowers = list(self._brands_by_lower.keys())
        self._unique_generic_lowers = list(generics_set)
        logger.info(
            "[DEV-DB] Loaded %d medicine records (%d unique brand names, %d generics) from %s",
            len(self._brands_by_id),
            len(self._unique_brand_lowers),
            len(self._unique_generic_lowers),
            self.db_path.name,
        )

    def is_generic_ingredient_word(self, word_lower: str) -> bool:
        """Returns True if `word_lower` is a generic active ingredient name (e.g. 'clonazepam', 'ondansetron', 'naproxen')."""
        if word_lower in self._unique_generic_lowers or word_lower in self._generic_first_words:
            return True
        return False

    def get_full_medicine_info(self, brand_id: int) -> Optional[MedicineInfo]:
        if not self.db_path.exists():
            return None
        with sqlite3.connect(f"{self.db_path.as_uri()}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM medicines WHERE brand_id = ?",
                (brand_id,),
            ).fetchone()
            if not row:
                return None
            keys = row.keys()
            return MedicineInfo(
                brand_id=int(row["brand_id"]),
                brand_name=row["brand_name"],
                generic=row["generic"],
                dosage_form=row["dosage_form"],
                manufacturer=row["manufacturer"],
                package_container=row["package_container"],
                drug_class=row["drug_class"],
                indication=row["indication"],
                indication_description=row["indication_description"],
                dosage_description=row["dosage_description"],
                interaction_description=row["interaction_description"],
                contraindications_description=row["contraindications_description"],
                side_effects_description=row["side_effects_description"],
                pregnancy_and_lactation_description=row["pregnancy_and_lactation_description"],
                precautions_description=row["precautions_description"],
                storage_conditions_description=row["storage_conditions_description"],
                beginner_summary=row["beginner_summary"] if "beginner_summary" in keys else None,
            )

    def _pick_best_variant(
        self,
        entries: List[_BrandIndexEntry],
        preferred_dosage_form: Optional[str],
        extracted_strength: Optional[str],
    ) -> _BrandIndexEntry:
        if len(entries) == 1:
            return entries[0]

        if preferred_dosage_form:
            pref_lower = preferred_dosage_form.lower()
            for e in entries:
                if e.dosage_form and e.dosage_form.lower() == pref_lower:
                    return e
            for e in entries:
                if e.dosage_form and pref_lower in e.dosage_form.lower():
                    return e

        if extracted_strength:
            s_lower = extracted_strength.lower()
            for e in entries:
                if e.package_container and s_lower in e.package_container.lower():
                    return e

        # Prefer standard Tablet or Capsule for blister strips
        for exact_form in ("tablet", "capsule"):
            for e in entries:
                if e.dosage_form and e.dosage_form.lower() == exact_form:
                    return e
        for partial_form in ("tablet", "capsule"):
            for e in entries:
                if e.dosage_form and partial_form in e.dosage_form.lower():
                    return e

        return entries[0]

    def resolve_brand_from_query(self, query: str, min_score: float = 82.0) -> Optional[Tuple[str, float, str]]:
        """
        Matches a single cleaned query string against `_brands_by_lower`.
        Returns `(matched_brand_lower, score, method)` or `None`.
        """
        q_low = query.strip().lower()
        if len(q_low) < 3:
            return None

        if q_low in self._brands_by_lower:
            return (q_low, 100.0, "exact_brand")

        # Only fuzzy-match queries of length >= 4 to avoid matching 3-letter noise fragments
        if len(q_low) < 4 or not self._unique_brand_lowers:
            return None

        hit = process.extractOne(
            q_low,
            self._unique_brand_lowers,
            scorer=fuzz.WRatio,
            score_cutoff=min_score,
        )
        if hit:
            matched_lower, score, _ = hit
            # Guard against huge length mismatch (e.g. 4-char noise matching a 14-char brand via partial WRatio)
            len_ratio = min(len(q_low), len(matched_lower)) / max(1, max(len(q_low), len(matched_lower)))
            if len_ratio < 0.65 and score < 95.0:
                return None
            return (matched_lower, round(float(score), 1), "fuzzy_brand")
        return None

    def _build_candidate_queries(self, raw_text: str, ocr_lines: Optional[List[OCRLine]] = None) -> List[str]:
        candidates: List[str] = []

        if ocr_lines:
            sorted_lines = sorted(ocr_lines, key=lambda l: (l.confidence, l.box_height), reverse=True)
            for ln in sorted_lines:
                cleaned = strip_strengths_and_noise(ln.text)
                if cleaned and cleaned.lower() not in [c.lower() for c in candidates]:
                    candidates.append(cleaned)

        for raw_line in raw_text.splitlines():
            cleaned = strip_strengths_and_noise(raw_line)
            if cleaned and cleaned.lower() not in [c.lower() for c in candidates]:
                candidates.append(cleaned)

        full_cleaned = strip_strengths_and_noise(raw_text)
        words = full_cleaned.split()

        for n in (3, 2, 1):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i : i + n])
                if len(ngram) >= 3 and ngram.lower() not in [c.lower() for c in candidates]:
                    candidates.append(ngram)

        return candidates

    def match_strip_text(
        self,
        raw_text: str,
        ocr_lines: Optional[List[OCRLine]] = None,
        strip_index: int = 1,
        bbox: Optional[BoundingBox] = None,
        forced_brand_lower: Optional[str] = None,
        forced_score: float = 100.0,
        forced_method: str = "exact_brand",
    ) -> StripScanResult:
        dosage_form = detect_dosage_form(raw_text)
        candidates = self._build_candidate_queries(raw_text, ocr_lines)

        best_brand_lower: Optional[str] = forced_brand_lower
        best_query: Optional[str] = forced_brand_lower
        best_score: float = forced_score if forced_brand_lower else 0.0
        match_method: Optional[str] = forced_method if forced_brand_lower else None

        # Pass 1: Exact brand_lower match (preferring brand names over generic ingredient names)
        if best_brand_lower is None:
            exact_brand_only: List[Tuple[int, str, str]] = []
            exact_any: List[Tuple[int, str, str]] = []
            for q in candidates:
                q_low = q.lower()
                if q_low in self._brands_by_lower:
                    word_count = len(q_low.split())
                    exact_any.append((word_count, q, q_low))
                    if not self.is_generic_ingredient_word(q_low):
                        exact_brand_only.append((word_count, q, q_low))

            chosen_pool = exact_brand_only if exact_brand_only else exact_any
            if chosen_pool:
                chosen_pool.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
                _, best_query, best_brand_lower = chosen_pool[0]
                best_score = 100.0
                match_method = "exact_brand"

        # Pass 2: RapidFuzz fuzzy match across unique brand names
        if best_brand_lower is None and self._unique_brand_lowers:
            for q in candidates:
                q_low = q.lower()
                if len(q_low) < 4 or self.is_generic_ingredient_word(q_low):
                    continue
                res = self.resolve_brand_from_query(q_low, min_score=MIN_FUZZY_SCORE)
                if res:
                    matched_lower, score, method = res
                    len_ratio = min(len(q_low), len(matched_lower)) / max(len(q_low), len(matched_lower))
                    adjusted_score = float(score) * (0.85 + 0.15 * len_ratio)
                    if adjusted_score > best_score:
                        best_score = score
                        best_brand_lower = matched_lower
                        best_query = q
                        match_method = method

        matched_medicine: Optional[MedicineInfo] = None
        alternatives: List[MedicineInfo] = []

        # Extract strength using the resolved brand name as context (helps with 'Prelica®50', 'Epitra°1', etc.)
        brand_hint = self._brands_by_lower[best_brand_lower][0].brand_name if best_brand_lower else None
        primary_strength, all_strengths = extract_strengths(raw_text, brand_name=brand_hint)

        if best_brand_lower is not None:
            entries = self._brands_by_lower[best_brand_lower]
            chosen = self._pick_best_variant(entries, dosage_form, primary_strength)
            matched_medicine = self.get_full_medicine_info(chosen.brand_id)
            for alt in entries:
                if alt.brand_id != chosen.brand_id and len(alternatives) < 3:
                    info = self.get_full_medicine_info(alt.brand_id)
                    if info:
                        alternatives.append(info)
        elif self._unique_generic_lowers:
            for q in candidates:
                q_low = q.lower()
                if len(q_low) < 4:
                    continue
                hit = process.extractOne(
                    q_low,
                    self._unique_generic_lowers,
                    scorer=fuzz.WRatio,
                    score_cutoff=82.0,
                )
                if hit:
                    matched_gen_lower, score, _ = hit
                    with sqlite3.connect(f"{self.db_path.as_uri()}?mode=ro", uri=True) as conn:
                        row = conn.execute(
                            "SELECT brand_id FROM medicines WHERE generic_lower = ? LIMIT 1",
                            (matched_gen_lower,),
                        ).fetchone()
                        if row:
                            matched_medicine = self.get_full_medicine_info(int(row[0]))
                            best_score = round(float(score), 1)
                            best_query = q
                            match_method = "generic_fallback"
                            break

        logger.info(
            "[DEV-FUZZ] Strip #%d -> Brand=%r | Strength=%r | Generic=%r | Score=%.1f%% (%s)",
            strip_index,
            matched_medicine.brand_name if matched_medicine else None,
            primary_strength,
            matched_medicine.generic if matched_medicine else None,
            best_score,
            match_method,
        )

        return StripScanResult(
            strip_index=strip_index,
            bbox=bbox,
            raw_ocr_text=raw_text.strip(),
            ocr_lines=ocr_lines or [],
            extracted_brand_query=best_query,
            extracted_strength=primary_strength,
            all_detected_strengths=all_strengths,
            detected_dosage_form=dosage_form or (matched_medicine.dosage_form if matched_medicine else None),
            match_score=best_score,
            match_method=match_method,
            matched_medicine=matched_medicine,
            alternatives=alternatives,
        )

    def search_medicines(self, query: str, limit: int = 10) -> List[MedicineInfo]:
        q = query.strip().lower()
        if not q or not self.db_path.exists():
            return []

        results: List[MedicineInfo] = []
        seen_ids = set()

        with sqlite3.connect(f"{self.db_path.as_uri()}?mode=ro", uri=True) as conn:
            rows = conn.execute(
                """
                SELECT brand_id FROM brands
                WHERE brand_lower LIKE ? OR generic_lower LIKE ?
                ORDER BY CASE WHEN brand_lower = ? THEN 0 WHEN brand_lower LIKE ? THEN 1 ELSE 2 END, brand_name ASC
                LIMIT ?
                """,
                (f"%{q}%", f"%{q}%", q, f"{q}%", limit),
            ).fetchall()
            for (bid,) in rows:
                bid_int = int(bid)
                if bid_int not in seen_ids:
                    seen_ids.add(bid_int)
                    info = self.get_full_medicine_info(bid_int)
                    if info:
                        results.append(info)

        return results


# ---------------------------------------------------------------------------
# 4. Multi-Strip Duplicate & Drug-Interaction Safety Checker
# ---------------------------------------------------------------------------


def _split_active_ingredients(generic_name: str) -> List[str]:
    return [part.strip().lower() for part in re.split(r"\+|/|&|,", generic_name) if part.strip()]


def evaluate_multi_strip_safety(strips: Sequence[StripScanResult]) -> Tuple[List[SafetyAlert], str, str]:
    alerts: List[SafetyAlert] = []
    matched_strips = [s for s in strips if s.matched_medicine is not None]

    ingredient_map: Dict[str, List[StripScanResult]] = {}
    for s in matched_strips:
        med = s.matched_medicine
        assert med is not None
        for ing in _split_active_ingredients(med.generic):
            ingredient_map.setdefault(ing, []).append(s)

    for ing, group in ingredient_map.items():
        if len(group) > 1:
            strip_ids = [s.strip_index for s in group]
            brands = [
                f"{s.matched_medicine.brand_name}{f' ({s.extracted_strength})' if s.extracted_strength else ''}"
                for s in group
                if s.matched_medicine
            ]
            ing_title = ing.title()
            alerts.append(
                SafetyAlert(
                    alert_type="duplicate_generic",
                    severity="high",
                    involved_strips=strip_ids,
                    involved_brands=brands,
                    generic_or_class=ing_title,
                    message_en=(
                        f"Duplicate Active Ingredient Alert: {', '.join(brands)} all contain {ing_title}. "
                        f"Taking them together may cause accidental overdose."
                    ),
                    message_bn=(
                        f"সতর্কতা (একই উপাদান): {', '.join(brands)} ওষুধগুলোতে একই সক্রিয় উপাদান ({ing_title}) রয়েছে। "
                        f"এগুলো একসাথে খেলে অতিরিক্ত মাত্রার (Overdose) ঝুঁকি হতে পারে।"
                    ),
                )
            )

    for i in range(len(matched_strips)):
        for j in range(i + 1, len(matched_strips)):
            s1, s2 = matched_strips[i], matched_strips[j]
            m1, m2 = s1.matched_medicine, s2.matched_medicine
            assert m1 is not None and m2 is not None

            b1_label = f"{m1.brand_name}{f' ({s1.extracted_strength})' if s1.extracted_strength else ''}"
            b2_label = f"{m2.brand_name}{f' ({s2.extracted_strength})' if s2.extracted_strength else ''}"

            ings1 = _split_active_ingredients(m1.generic)
            ings2 = _split_active_ingredients(m2.generic)

            if set(ings1) & set(ings2):
                continue

            inter1 = (m1.interaction_description or "").lower()
            inter2 = (m2.interaction_description or "").lower()

            interacting_ing: Optional[str] = None
            for ing in ings2:
                if len(ing) >= 4 and re.search(rf"\b{re.escape(ing)}\b", inter1):
                    interacting_ing = ing.title()
                    break
            if not interacting_ing:
                for ing in ings1:
                    if len(ing) >= 4 and re.search(rf"\b{re.escape(ing)}\b", inter2):
                        interacting_ing = ing.title()
                        break

            if interacting_ing:
                alerts.append(
                    SafetyAlert(
                        alert_type="drug_interaction",
                        severity="high",
                        involved_strips=[s1.strip_index, s2.strip_index],
                        involved_brands=[b1_label, b2_label],
                        generic_or_class=f"{m1.generic} <-> {m2.generic}",
                        message_en=(
                            f"Potential Drug Interaction between {b1_label} ({m1.generic}) and "
                            f"{b2_label} ({m2.generic}). Consult a physician before taking together."
                        ),
                        message_bn=(
                            f"ওষুধের আন্তঃক্রিয়া সতর্কতা: {b1_label} ({m1.generic}) এবং {b2_label} ({m2.generic}) "
                            f"একসাথে সেবনে পার্শ্বপ্রতিক্রিয়া হতে পারে। চিকিৎসকের পরামর্শ নিন।"
                        ),
                    )
                )
            elif m1.drug_class and m2.drug_class and m1.drug_class.lower() == m2.drug_class.lower():
                alerts.append(
                    SafetyAlert(
                        alert_type="same_drug_class",
                        severity="moderate",
                        involved_strips=[s1.strip_index, s2.strip_index],
                        involved_brands=[b1_label, b2_label],
                        generic_or_class=m1.drug_class,
                        message_en=(
                            f"Same Therapeutic Class: {b1_label} and {b2_label} both belong to '{m1.drug_class}'."
                        ),
                        message_bn=(
                            f"একই শ্রেণীর ওষুধ: {b1_label} এবং {b2_label} উভয়ই '{m1.drug_class}' শ্রেণীর অন্তর্ভুক্ত।"
                        ),
                    )
                )

    if not matched_strips:
        summary_en = "No matching medicines could be identified from the scan. Please retake a clearer photo."
        summary_bn = "ছবি থেকে কোনো ওষুধ শনাক্ত করা যায়নি। অনুগ্রহ করে স্পষ্ট আলোতে আবার ছবি তুলুন।"
    else:
        items_en = []
        items_bn = []
        for s in matched_strips:
            m = s.matched_medicine
            assert m is not None
            strength_str = f" {s.extracted_strength}" if s.extracted_strength else ""
            form_str = f" ({s.detected_dosage_form})" if s.detected_dosage_form else ""
            items_en.append(f"Strip {s.strip_index}: {m.brand_name}{strength_str}{form_str} — Generic: {m.generic}")
            items_bn.append(
                f"স্ট্রিপ {s.strip_index}: {m.brand_name}{strength_str}{form_str} — জেনেরিক: {m.generic}"
            )

        summary_en = " | ".join(items_en)
        summary_bn = " | ".join(items_bn)
        if alerts:
            summary_en += " || WARNINGS: " + " ".join(a.message_en for a in alerts)
            summary_bn += " || সতর্কতা: " + " ".join(a.message_bn for a in alerts)
        else:
            summary_bn += "। শনাক্তকৃত ওষুধে কোনো ডুপ্লিকেট বা ঝুঁকিপূর্ণ আন্তঃক্রিয়া পাওয়া যায়নি।"

    return alerts, summary_en, summary_bn


# ---------------------------------------------------------------------------
# 5. End-to-End Multi-Strip Scanner (RapidOCR Spatial Clustering + RapidFuzz)
# ---------------------------------------------------------------------------


def _box_center(b: Optional[BoundingBox], default_w: int = 720, default_h: int = 1280) -> Tuple[float, float]:
    if b is None:
        return (default_w / 2.0, default_h / 2.0)
    return ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0)


class MedicineStripScanner:
    """
    End-to-end scanner that:
    1. Runs RapidOCR (PP-OCRv6 ONNX) across the image (0° + 90° CCW) to detect all strip text & bounding boxes.
    2. Identifies distinct medicine brand anchors in the image and groups nearby OCR text boxes (strengths,
       generic subtitles, dosage forms) into per-strip spatial clusters (supporting 1 to 6 strips in a single photo).
    3. Extracts each strip's exact dosage strength ('500 mg', '50 mg', '10 mg', '8 mg', '1 mg', etc.)
       and matches full clinical info + `beginner_summary` from `data/processed/medicines.db`.
    4. Runs deterministic duplicate & drug-interaction safety checks.
    """

    def __init__(self, db_path: Path = DB_PATH, yolo_model_path: Optional[Path] = None) -> None:
        self.detector = StripDetector(model_path=yolo_model_path)
        self.ocr_engine = OCREngine()
        self.matcher = MedicineMatcher(db_path=db_path)

    def _cluster_ocr_lines_into_strips(
        self,
        ocr_lines: List[OCRLine],
        img_w: int,
        img_h: int,
    ) -> List[Tuple[str, float, str, List[OCRLine], BoundingBox]]:
        """
        Groups OCR lines from a full photo into 1..N distinct medicine strip clusters based on
        detected brand anchor boxes and their spatial positions `(cx, cy)`.
        """
        if not ocr_lines:
            return []

        # Step 1: Identify brand anchor detections among OCR lines
        # Map: brand_lower -> list of (line, score, method)
        brand_anchors: Dict[str, List[Tuple[OCRLine, float, str]]] = {}

        for ln in ocr_lines:
            if ln.confidence < 0.70:
                continue
            cleaned = strip_strengths_and_noise(ln.text)
            if not cleaned:
                continue

            # Check full cleaned line and individual words (length >= 5 for fuzzy, >= 4 for exact)
            tokens_to_check = [cleaned] + [w for w in cleaned.split() if len(w) >= 4]
            for tok in tokens_to_check:
                tok_low = tok.lower()
                if self.matcher.is_generic_ingredient_word(tok_low):
                    continue
                resolved = self.matcher.resolve_brand_from_query(tok_low, min_score=86.0)
                if resolved:
                    b_low, score, method = resolved
                    # Require high confidence for fuzzy single-token brand anchors so OCR fragments don't spawn ghost strips
                    if method != "exact_brand" and (score < 90.0 or len(tok_low) < 5):
                        continue
                    brand_anchors.setdefault(b_low, []).append((ln, score, method))
                    break

        # If we found >= 2 brand anchors, filter out low-score fuzzy ghosts when exact brand matches exist
        if len(brand_anchors) >= 2:
            exact_brands = {
                b_low
                for b_low, items in brand_anchors.items()
                if any(m == "exact_brand" and ln.confidence >= 0.85 for ln, _, m in items)
            }
            if exact_brands:
                brand_anchors = {b_low: brand_anchors[b_low] for b_low in exact_brands}

        if not brand_anchors:
            # Fallback: single strip containing all OCR lines
            xs = [ln.bbox.x1 for ln in ocr_lines if ln.bbox] + [ln.bbox.x2 for ln in ocr_lines if ln.bbox]
            ys = [ln.bbox.y1 for ln in ocr_lines if ln.bbox] + [ln.bbox.y2 for ln in ocr_lines if ln.bbox]
            bbox = BoundingBox(
                x1=max(0, min(xs) - 20) if xs else 0,
                y1=max(0, min(ys) - 20) if ys else 0,
                x2=min(img_w, max(xs) + 20) if xs else img_w,
                y2=min(img_h, max(ys) + 20) if ys else img_h,
                confidence=0.9,
            )
            return [("", 0.0, "auto", ocr_lines, bbox)]

        # Step 2: Compute each brand cluster's spatial centroid (cx, cy)
        cluster_centers: Dict[str, Tuple[float, float]] = {}
        cluster_meta: Dict[str, Tuple[float, str]] = {}
        cluster_lines: Dict[str, List[OCRLine]] = {b_low: [] for b_low in brand_anchors}

        for b_low, anchor_list in brand_anchors.items():
            cxs = [_box_center(ln.bbox, img_w, img_h)[0] for ln, _, _ in anchor_list]
            cys = [_box_center(ln.bbox, img_w, img_h)[1] for ln, _, _ in anchor_list]
            cluster_centers[b_low] = (sum(cxs) / len(cxs), sum(cys) / len(cys))
            best_score = max(sc for _, sc, _ in anchor_list)
            best_method = "exact_brand" if any(m == "exact_brand" for _, _, m in anchor_list) else "fuzzy_brand"
            cluster_meta[b_low] = (best_score, best_method)

        # Step 3: Assign every OCR line in the image to its spatially nearest brand cluster
        anchor_line_owner: Dict[int, str] = {}
        for b_low, anchor_list in brand_anchors.items():
            for ln, _, _ in anchor_list:
                anchor_line_owner[id(ln)] = b_low

        for ln in ocr_lines:
            if id(ln) in anchor_line_owner:
                cluster_lines[anchor_line_owner[id(ln)]].append(ln)
                continue
            lx, ly = _box_center(ln.bbox, img_w, img_h)
            nearest_brand = min(
                cluster_centers.keys(),
                key=lambda b_low: math.hypot(lx - cluster_centers[b_low][0], ly - cluster_centers[b_low][1]),
            )
            cluster_lines[nearest_brand].append(ln)

        # Step 4: Sort clusters left-to-right, top-to-bottom and compute strip bounding box
        ordered_brands = sorted(
            cluster_centers.keys(),
            key=lambda b_low: (round(cluster_centers[b_low][0] / max(1, img_w * 0.25)), cluster_centers[b_low][1]),
        )

        clusters_out: List[Tuple[str, float, str, List[OCRLine], BoundingBox]] = []
        pad_x = int(img_w * 0.04)
        pad_y = int(img_h * 0.04)

        for b_low in ordered_brands:
            lns = cluster_lines[b_low]
            xs = [ln.bbox.x1 for ln in lns if ln.bbox] + [ln.bbox.x2 for ln in lns if ln.bbox]
            ys = [ln.bbox.y1 for ln in lns if ln.bbox] + [ln.bbox.y2 for ln in lns if ln.bbox]
            conf = round(max((ln.confidence for ln in lns), default=0.95), 3)
            bbox = BoundingBox(
                x1=max(0, (min(xs) if xs else 0) - pad_x),
                y1=max(0, (min(ys) if ys else 0) - pad_y),
                x2=min(img_w, (max(xs) if xs else img_w) + pad_x),
                y2=min(img_h, (max(ys) if ys else img_h) + pad_y),
                confidence=conf,
            )
            sc, meth = cluster_meta[b_low]
            clusters_out.append((b_low, sc, meth, lns, bbox))

        return clusters_out

    def scan_image(self, image: np.ndarray) -> MultiStripScanResponse:
        if image is None or image.size == 0:
            return MultiStripScanResponse(
                total_strips_detected=0,
                strips=[],
                safety_alerts=[],
                summary_en="Invalid or empty image.",
                summary_bn="ছবিটি পড়া যায়নি।",
            )

        h, w = image.shape[:2]
        logger.info("[DEV-SCAN] Starting multi-strip image scan (%dx%d)", w, h)

        # Run RapidOCR on the original image (plus 90° CCW rotation for vertical strip text)
        ocr_lines = self.ocr_engine.extract_lines(image, include_rotated=True)

        # If raw image had very low contrast, also try CLAHE-enhanced image
        if not ocr_lines:
            enhanced = self.detector.preprocess_for_ocr(image)
            ocr_lines = self.ocr_engine.extract_lines(enhanced, include_rotated=True)

        clusters = self._cluster_ocr_lines_into_strips(ocr_lines, img_w=w, img_h=h)
        strip_results: List[StripScanResult] = []

        for idx, (b_low, sc, meth, lns, bbox) in enumerate(clusters, start=1):
            raw_text = "\n".join(line.text for line in lns)
            result = self.matcher.match_strip_text(
                raw_text=raw_text,
                ocr_lines=lns,
                strip_index=idx,
                bbox=bbox,
                forced_brand_lower=b_low if b_low else None,
                forced_score=sc,
                forced_method=meth,
            )
            strip_results.append(result)

        alerts, summary_en, summary_bn = evaluate_multi_strip_safety(strip_results)
        logger.info(
            "[DEV-SCAN] Completed scan: %d strip(s) detected, %d safety alert(s)",
            len(strip_results),
            len(alerts),
        )
        return MultiStripScanResponse(
            total_strips_detected=len(strip_results),
            strips=strip_results,
            safety_alerts=alerts,
            summary_en=summary_en,
            summary_bn=summary_bn,
        )

    def scan_texts(self, strip_texts: Sequence[str]) -> MultiStripScanResponse:
        strip_results: List[StripScanResult] = []
        for idx, txt in enumerate(strip_texts, start=1):
            lines = [OCRLine(text=ln.strip(), confidence=1.0, box_height=20.0) for ln in txt.splitlines() if ln.strip()]
            res = self.matcher.match_strip_text(
                raw_text=txt,
                ocr_lines=lines,
                strip_index=idx,
            )
            strip_results.append(res)

        alerts, summary_en, summary_bn = evaluate_multi_strip_safety(strip_results)
        return MultiStripScanResponse(
            total_strips_detected=len(strip_results),
            strips=strip_results,
            safety_alerts=alerts,
            summary_en=summary_en,
            summary_bn=summary_bn,
        )
