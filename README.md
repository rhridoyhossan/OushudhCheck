# OushudhCheck: Multi-Strip Medicine Safety Scanner & Bangla Voice Assistant

> **DualStack**: *Scans several medicine strips in one photo, flags duplicate ingredients and interactions, and speaks dosage guidance in Bangla.*

[![Platform](https://img.shields.io/badge/Platform-Next.js_16_%7C_FastAPI_%7C_Telegram_Bot-0b5563?style=for-the-badge)](#system-architecture--end-to-end-data-flow)
[![Vision & OCR](https://img.shields.io/badge/Vision-YOLO26s_ONNX_%2B_PP--OCRv6_%2B_CLAHE-0ea86b?style=for-the-badge)](#challenge-2--trustworthy-safety-alerts-on-blurry-photos-unreadable-brands--similar-generic-names)
[![Safety SLM](https://img.shields.io/badge/Safety_SLM-Fine--Tuned_Gemma_3_1B_Q4__K__M_GGUF-7c3aed?style=for-the-badge)](#architectural-evolution-proposal-cintestpdf-vs-upgraded-production-system)
[![Database](https://img.shields.io/badge/BD_Medicine_DB-17%2C191_Brands_%7C_1%2C661_Generics-2563eb?style=for-the-badge)](#4-dataset-engineering--sqlite-fts5-ingestion-make_datasetipynb--scriptsingest_drugspy)
[![Offline & Cost](https://img.shields.io/badge/100%25_Offline_CPU-%240_API_Cost_%7C_100%25_Private-d97706?style=for-the-badge)](#why-oushudhcheck-matters--cost-privacy-offline-autonomy--real-world-use-cases-in-bangladesh)

---

## Executive Overview

In Bangladesh, patients and elderly caregivers frequently keep loose medicine blister strips at home from multiple prescriptions and local pharmacy visits without outer cartons or written instructions. Because over **240 pharmaceutical companies** sell identical active generic compounds under dozens of different brand names—for example, **Napa**, **Ace**, **Renova**, and **Napa Extra** all contain **Paracetamol**—families routinely take duplicate active ingredients at the same time without realizing it, risking accidental overdose and severe drug-drug interactions (DDI). Compounding this risk, medicine strips print critical generic names and strengths (`mg`, `mcg`, `ml`, `IU`) in tiny English typography on specular, reflective aluminum back-foil that elderly, rural, and low-literacy patients struggle to read.

**OushudhCheck** turns any smartphone camera or Telegram chat (`@OushudhCheck_bot`) into an offline-capable, zero-cost **Multi-Strip Medicine Safety Scanner & Bangla Voice Medication Assistant**. Users place **1 to 6 medicine blister strips** on a table, capture a single photo, and immediately receive:

1. **Multi-Strip Detection & Reflective Foil OCR**: Isolates each physical strip and reads tiny, rotated, or glare-heavy foil text using **YOLO26s ONNX Hybrid Fusion**, **OpenCV LAB-CLAHE** contrast enhancement, and dual-orientation **PaddleOCR PP-OCRv6 (`0°` + `90° CCW`)**.
2. **Verified Brand-to-Generic Resolution**: Resolves noisy OCR tokens against a curated, read-only local SQLite database ([`data/processed/medicines.db`](data/processed/medicines.db)) containing **17,191 verified Bangladeshi medicine brands** and **1,661 unique generic formulations** across **400+ therapeutic drug classes** (feature-engineered and deduplicated from **21,714 raw Kaggle/MedEx Bangladesh records**), including **EDCL (Essential Drugs Co. Ltd.)** government generic strips.
3. **100% Deterministic Duplicate & Drug-Interaction Alerts**: Decomposes single and multi-ingredient combination strips (`+`, `/`, `&`, `,`) into active generic compounds to flag duplicate active ingredients, pairwise DDI risks, and overlapping therapeutic classes with mathematical certainty.
4. **Grounded Bangla Clinical Guidance, Daily Schedule & Neural Voice**: Uses our fine-tuned **`Gemma 3 1B-Q4_K_M` Bangla Safety SLM** ([`models/oushudhcheck-gemma3-1b-bn.Q4_K_M.gguf`](models/system_prompt.txt)) and deterministic clinical grounding engine ([`backend/app/services/slm.py`](backend/app/services/slm.py)) to generate clear Bangla explanations (`সহজ বাংলা সারসংক্ষেপ`), Morning/Noon/Night meal schedules, and natural spoken Bangla audio (`edge-tts` `bn-BD-NabanitaNeural`)—plus stateful **Caregiver Mode (`/caregiver`)** for Bangla voice follow-up Q&A on Telegram.

### Key Performance & Impact Metrics

| Evaluation Benchmark (`project.pdf` & Onsite Final) | Target Goal | Achieved Production Performance | Verification Source |
| :--- | :---: | :---: | :--- |
| **Brand-to-Generic Resolution Accuracy** | `> 92%` | **100% on 6 real multi-strip tabletop benchmark photos (15/15 strips)** & **>95%** across noisy/glare/rotated foil tests | [`scripts/benchmark.py`](scripts/benchmark.py) · [`backend/app/api/routes.py`](backend/app/api/routes.py) |
| **Duplicate Active Generic Detection** | `100% Deterministic` | **100% Deterministic** across single & combination strips (`Napa` + `Ace`, `Napa` + `Napa Extra`) with **0% false alarms** on distinct generics | [`evaluate_multi_strip_safety()`](backend/app/services/ocr.py) · [`tests/test_caregiver_qa.py`](tests/test_caregiver_qa.py) |
| **End-to-End Bangla Voice Turnaround** | `< 6 seconds` | **~1.8s – 4.5s** full pipeline scan + atomic SHA-256 cached **`bn-BD-NabanitaNeural`** voice synthesis | [`backend/app/services/voice.py`](backend/app/services/voice.py) · [`/api/v1/scan`](backend/app/api/routes.py) |
| **Bangladeshi Pharmaceutical Coverage** | `21,000+ raw records` | **17,191 deduplicated BD brands · 1,661 unique generics · 400+ drug classes · 240+ manufacturers + EDCL virtual generics** | [`scripts/ingest_drugs.py`](scripts/ingest_drugs.py) · [`data/processed/medicines.db`](data/processed/medicines.db) |
| **Caregiver Mode (`/caregiver`) Follow-Up Q&A** | `100% Bangla Voice/Text Q&A` | **9 clinical intents · 30-min stateful session memory · 100% pure Bangla question-specific answers + voice reply** | [`answer_caregiver_followup()`](backend/app/services/slm.py) · [`bot/telegram_bot.py`](bot/telegram_bot.py) |
| **Runtime Cost & Cloud Dependency** | `Edge-efficient local AI` | **$0.00 / scan · 100% Offline CPU capable · Unlimited runs · Bring-Your-Own-Model (`SLM_API_URL`)** | [`docker-compose.yml`](docker-compose.yml) · [`.env.example`](.env.example) |

---

## Table of Contents

1. [Executive Overview & Key Metrics](#executive-overview)
2. [Onsite Challenges Solved (100% Complete & Verified)](#onsite-challenges-solved-100-complete--verified)
   - [Challenge 1: Caregiver Mode (`/caregiver`) with Bangla Voice Follow-Up Q&A on Telegram](#challenge-1--caregiver-mode-caregiver-with-bangla-voice-follow-up-qa-on-telegram)
   - [Challenge 2: Trustworthy Safety Alerts on Blurry Photos, Unreadable Brands & Similar Generic Names](#challenge-2--trustworthy-safety-alerts-on-blurry-photos-unreadable-brands--similar-generic-names)
3. [Architectural Evolution: Proposal (`project.pdf`) vs. Upgraded Production System](#architectural-evolution-proposal-cintestpdf-vs-upgraded-production-system)
4. [System Architecture & End-to-End Data Flow](#system-architecture--end-to-end-data-flow)
5. [Why OushudhCheck Matters: Cost, Privacy, Offline Autonomy & Real-World Use Cases in Bangladesh](#why-oushudhcheck-matters--cost-privacy-offline-autonomy--real-world-use-cases-in-bangladesh)
6. [Complete Model Training & Dataset Engineering Pipeline (`ml_training/` & `scripts/`)](#complete-model-training--dataset-engineering-pipeline-ml_training--scripts)
7. [Complete REST API Reference](#complete-rest-api-reference)
8. [Test & Benchmark Suite (`test/` & `tests/`)](#test--benchmark-suite-test--tests)
9. [How to Run, Deploy & Configure Custom GGUF Models](#how-to-run-deploy--configure-custom-gguf-models)
10. [Repository Structure & Team DualStack](#repository-structure)

---

## Onsite Challenges Solved (100% Complete & Verified)

### Challenge 1: Caregiver Mode (`/caregiver`) with Bangla Voice Follow-Up Q&A on Telegram

Family caregivers and elderly patients often need specific follow-up answers after scanning a medicine strip—such as *"Should this be taken before or after meals?"*, *"Can it be taken in the morning?"*, *"What are the side effects?"*, or *"Is it safe during pregnancy?"*—without wading through a full medicine monograph or reading English text.

We engineered a complete, stateful **Caregiver Mode (`/caregiver`)** in [`bot/telegram_bot.py`](bot/telegram_bot.py) backed by [`answer_caregiver_followup()`](backend/app/services/slm.py) and [`backend/app/services/voice.py`](backend/app/services/voice.py):

```mermaid
sequenceDiagram
    autonumber
    actor CG as Caregiver / Elderly User
    participant Bot as @OushudhCheck_bot (aiohttp)
    participant Scan as MedicineStripScanner (YOLO26s + PP-OCRv6)
    participant ASR as Bangla ASR (bn-BD + faster-whisper int8)
    participant QA as answer_caregiver_followup (Gemma 3 1B + Grounded NLG)
    participant TTS as Edge Neural TTS (bn-BD-NabanitaNeural)

    CG->>Bot: /caregiver (or direct photo upload of 1–6 strips)
    Bot->>Scan: Run 5-stage live progress scan (editMessageText)
    Scan-->>Bot: Matched StripScanResult[] + SafetyAlert[]
    Bot->>Bot: Save to per-chat session memory (TTL = 30 mins / 1800s)
    Bot-->>CG: Full Bangla Scan Report + Safety Alerts + Spoken Bangla MP3
    CG->>Bot: Sends Bangla Voice Message (.ogg/.mp3) or Text Follow-Up Question
    Bot->>ASR: transcribe_bangla_audio() via PyAV 16kHz mono
    ASR-->>Bot: Transcribed Bangla / Banglish question
    Bot->>QA: answer_caregiver_followup(question, session_strips, session_alerts)
    QA->>QA: Detect intent(s) + focus strip(s) + enforce route & clinical grounding
    QA-->>Bot: Concise 1–2 sentence 100% Bangla answer (zero English leaks)
    Bot->>TTS: synthesize_bangla_speech(answer_bn)
    TTS-->>Bot: Cached / synthesized Bangla MP3 audio
    Bot-->>CG: Focused Bangla Text Answer + Spoken Bangla Voice Reply
```

#### Key Engineering Highlights of Caregiver Mode

1. **Stateful 30-Minute Session Memory (`CAREGIVER_SESSION_TTL_SEC = 1800`)**:
   - In [`bot/telegram_bot.py`](bot/telegram_bot.py), every photo scan automatically caches the detected `StripScanResult` list and `SafetyAlert` list in thread-safe per-chat memory (`self.caregiver_sessions[chat_id]`).
   - Users can explicitly toggle Caregiver Mode via `/caregiver` (and exit anytime via `/caregiver off` or `/exit`), **or** simply send a natural follow-up voice/text question within 30 minutes of any photo scan—`_looks_like_followup_question()` automatically detects follow-up utterances and routes them to `handle_caregiver_followup()`.
2. **Dual-Engine Bangla Voice Transcription (`transcribe_bangla_audio`)**:
   - Implemented in [`backend/app/services/voice.py`](backend/app/services/voice.py), incoming Telegram `.ogg`, `.mp3`, `.wav`, or `.m4a` voice clips are decoded directly into 16 kHz mono `float32` PCM arrays via `PyAV` (`_decode_audio_to_16k_mono()`), avoiding FFmpeg subprocess overhead and PyAV v19 API deprecations.
   - Uses a fast `bn-BD` speech recognizer as the primary pass for mixed Bangla + English medicine names, with automatic offline fallback to local **`faster-whisper` (`base`, `compute_type="int8"` on CPU)** featuring Voice Activity Detection (VAD), non-Bangla script filtering (`_NON_BANGLA_INDIC_RE`), and hallucination/repetition suppression (`_collect_valid_segments()`).
3. **9-Intent Clinical Question Routing & Multi-Strip Focus (`answer_caregiver_followup`)**:
   - Implemented in [`backend/app/services/slm.py`](backend/app/services/slm.py), `_detect_caregiver_intents()` classifies Bangla, Banglish, and English questions across 9 clinical intents:
     - `meal_timing` (*খাবারের আগে না পরে / খালি পেটে / ভরা পেটে / সকালে না রাতে*)
     - `dosage` (*দিনে কতবার / মাত্রা / ডোজ / কীভাবে ব্যবহার করব*)
     - `side_effects` (*পার্শ্বপ্রতিক্রিয়া / ঘুম আসে কি / মাথা ঘোরা / ক্ষতি*)
     - `pregnancy_lactation` (*গর্ভাবস্থায় / গর্ভবতী / স্তন্যদানকালে / বুকের দুধ*)
     - `interactions_duplicates` (*অন্য ওষুধের সাথে / একসাথে খেলে / বিক্রিয়া / ডুপ্লিকেট*)
     - `warnings_precautions` (*সতর্কতা / নিষেধাজ্ঞা / কিডনি / লিভার / হাঁপানি / গাড়ি চালানো*)
     - `storage` (*সংরক্ষণ / কোথায় রাখব / ফ্রিজে / তাপমাত্রা*)
     - `purpose` (*কী কাজ করে / কীসের ওষুধ / কোন রোগের জন্য*)
     - `ingredients` (*জেনেরিক উপাদান / কী দিয়ে তৈরি / মাত্রা কত*)
   - **Smart Multi-Strip Focus (`_select_focused_strips`)**: When a caregiver scans 4–5 strips at once and asks about one specific medicine (even with minor voice-ASR spelling drift like `"seklo"` for `"Seclo"`, matched at `fuzz.ratio >= 84`), the engine answers specifically for that strip while keeping cross-strip interaction alerts in view if the question asks about interactions or taking medicines together (`একসাথে`, `সবগুলো`, `উভয়`).
4. **100% Pure Bangla, Question-Specific Grounding Guardrails (`_is_slm_qa_answer_clinically_valid`)**:
   - Answers **only** what was asked in 1–2 concise sentences.
   - **Zero Boilerplate Regurgitation**: Rejects memorized summary openings (`_SUMMARY_BOILERPLATE_RE`: *"X হলো একটি ট্যাবলেট, যার মূল সক্রিয় উপাদান Y..."*) and blocks unsolicited ingredient or warning dumps when not asked.
   - **Zero English Prose Leaks (`_has_english_prose_leak`)**: Translates English `medicines.db` clinical descriptions into natural Bangla (via `_SIDE_EFFECT_MAP_BN`, `_INTERACTION_MAP_BN`, `_WARNING_MAP_BN`, `_extract_dosage_guidance_bn`, and `_extract_concise_purpose_bn`), preserving only exact medicine brand names, generic names, and dosage units (`mg`, `mcg`, `ml`, `IU`).
   - **Clinical Timing & Bedtime Guardrails**: Prevents dangerous timing contradictions—Proton Pump Inhibitors (`Omeprazole`, `Esomeprazole`, `Pantoprazole`, `Rabeprazole`, `Lansoprazole`) are always guided as **৩০ মিনিট আগে খালি পেটে** (30 minutes before meals on an empty stomach); NSAIDs (`Naproxen`, `Tolfenamic Acid`, `Diclofenac`, `Ibuprofen`, `Ketorolac`) and `Paracetamol` as **খাবারের পরে ভরা পেটে** (after meals on a full stomach); and sedatives/benzodiazepines (`Diazepam`, `Clonazepam`, `Alprazolam`, `Flunarizine`) as **রাতে ঘুমানোর আগে** (at night before sleep, explicitly warning against morning use due to drowsiness).
5. **Ready-to-Test Caregiver Voice Examples ([`test/caregiver_voice_examples/`](test/caregiver_voice_examples/README.txt))**:
   - We provide 4 pre-recorded Bangla caregiver voice clips in [`test/caregiver_voice_examples/`](test/caregiver_voice_examples/README.txt) ready to upload or forward directly to `@OushudhCheck_bot`:

| Audio File | Spoken Bangla Question | English Meaning | Target Intent |
| :--- | :--- | :--- | :--- |
| [`01_meal_timing.mp3`](test/caregiver_voice_examples/01_meal_timing.mp3) | *"এই ওষুধটি কি খাবারের আগে না পরে খাবে এবং দিনে কতবার দিতে হবে?"* | Should this medicine be taken before or after meals, and how many times a day? | `meal_timing` + `dosage` |
| [`02_side_effects.mp3`](test/caregiver_voice_examples/02_side_effects.mp3) | *"এই ওষুধ খেলে কী কী পার্শ্বপ্রতিক্রিয়া হতে পারে?"* | What side effects can happen from taking this medicine? | `side_effects` |
| [`03_pregnancy_safety.mp3`](test/caregiver_voice_examples/03_pregnancy_safety.mp3) | *"গর্ভাবস্থায় বা স্তন্যদানকালে কি এই ওষুধটি খাওয়া নিরাপদ?"* | Is it safe to take this medicine during pregnancy or breastfeeding? | `pregnancy_lactation` |
| [`04_interaction_warning.mp3`](test/caregiver_voice_examples/04_interaction_warning.mp3) | *"অন্য কোনো ওষুধের সাথে একসাথে খেলে কি কোনো সমস্যা বা বিক্রিয়া হতে পারে?"* | Can there be any problem or interaction if taken together with another medicine? | `interactions_duplicates` |

---

### Challenge 2: Trustworthy Safety Alerts on Blurry Photos, Unreadable Brands & Similar Generic Names

Real-world Bangladeshi blister strips pose severe computer vision and entity-resolution challenges: metallic aluminum back-foil produces harsh specular glare, half-used strips have torn edges that slice words in half (`"Amlodipine"` torn into `"dipine"`), multi-word brands split across lines (`"Napa®"` and `"Extra"`), government hospital strips (`EDCL`) have no commercial brand name at all, and non-oral medicines share generic compounds with oral tablets.

We solved every one of these failure modes through a layered, deterministic defence pipeline across [`backend/app/services/detector.py`](backend/app/services/detector.py), [`backend/app/services/ocr.py`](backend/app/services/ocr.py), and [`backend/app/services/slm.py`](backend/app/services/slm.py):

```mermaid
flowchart TD
    Img["Uploaded Multi-Strip Photo (1–6 Strips)"] --> Stage1["Stage 1: YOLO26s ONNX Strip Detection\n(Letterbox 640x640, end2end_xyxy6 NMS-free)"]
    Img --> Stage23["Stages 2 & 3: Dual-Orientation PP-OCRv6 (0° + 90° CCW)\n+ Lazy OpenCV LAB-CLAHE & Unsharp Mask"]
    Stage1 --> Stage4["Stage 4: 3-Stage YOLO26s + OCR Hybrid Spatial Fusion\n(yolo26_direct -> yolo26_crop_rescue -> yolo26_split -> ocr_fallback)"]
    Stage23 --> Stage4
    Stage4 --> Disambig["Stage 4b: Mini-AI Candidate Disambiguation (SafetySLMEngine)\n• Substring Ghost Suppression ('dipine' from 'Amlodipine')\n• Multi-Word Brand Upgrade ('Napa' + 'Extra' -> 'Napa Extra')\n• EDCL Government Strip & Generic Fallback Resolution\n• Context-Weighted Multi-Ingredient Strength Alignment"]
    Disambig --> Safety["Stage 5: Deterministic Multi-Ingredient Safety Graph\n• Split combination generics on + / & ,\n• 100% Duplicate Active Generic Alert\n• Pairwise Drug-Drug Interaction (DDI) & Class Overlap\n• Non-Oral Route Safety Guardrail (_enforce_route_safety)"]
    Safety --> Out["Verified Bangla Report + Safety Edges + Neural Voice Audio"]
```

#### 1. Blurry & Glare-Heavy Reflective Foil Strips
- **OpenCV LAB-CLAHE + Unsharp Mask (`preprocess_for_ocr_with_scale`)**: Converts BGR foil crops to CIE `LAB` color space, applies Contrast Limited Adaptive Histogram Equalization (`clipLimit=2.2, tileGridSize=(8,8)`) strictly on the luminance (`L`) channel to cut metallic glare without distorting ink color, applies Gaussian unsharp masking (`1.35× - 0.35×`), and cubically upscales small crops (`<640px`) while returning the exact `upscale_factor` so crop-space OCR boxes inverse-map back to original image pixels without coordinate drift ([`backend/app/services/detector.py`](backend/app/services/detector.py)).
- **Dual-Orientation PP-OCRv6 (`0°` + `90° CCW`)**: [`OCREngine.extract_lines()`](backend/app/services/ocr.py) runs RapidOCR (PP-OCRv6 ONNX) on both standard `0°` and rotated `90° counter-clockwise` frames, recovering vertically printed strips (such as `Naprosyn 500mg` in `test/img (2).jpeg`) and inversely mapping rotated coordinates back to the original image canvas.
- **3-Stage YOLO26s Hybrid Spatial Fusion**: [`MedicineStripScanner.run_stage4_fuse_and_match()`](backend/app/services/ocr.py) (detailed in [`docs/designs/yolo26-hybrid-strip-detector.md`](docs/designs/yolo26-hybrid-strip-detector.md)) fuses real **YOLO26s ONNX** bounding boxes with global OCR spatial-semantic clusters, tracking per-strip provenance (`yolo26_direct`, `yolo26_crop_rescue`, `yolo26_split`, `ocr_fallback`). If glare causes full-image OCR to miss a strip, lazy CLAHE enhancement automatically rescues the unreadable lines; if YOLO misses a partially occluded strip on a crowded tabletop, free residual OCR clustering recovers it automatically with zero partial-miss regression.

#### 2. Unreadable Brand Name Rescue & Multi-Ingredient Strength Alignment
- **Generic-Ingredient Fallback Resolution (`match_method="generic_fallback"`)**: When a strip's commercial brand logo is torn off, blurred, or washed out by flash glare, [`MedicineMatcher.match_strip_text()`](backend/app/services/ocr.py) falls back to matching the extracted generic ingredient text against all **1,661 unique generics** in `medicines.db` (`fuzz.WRatio >= 82.0`) and resolves the corresponding clinical record and safety profile.
- **EDCL (Essential Drugs Co. Ltd.) Government Strip Support**: Bangladeshi government hospital strips (such as `DIAZEPAM TABLETS BP 5 mg` and `VITAMIN B-COMPLEX` in `test/newimg (2).jpg` and `test/newimg (3).jpg`) carry no commercial brand name. [`MedicineMatcher._load_index()`](backend/app/services/ocr.py) registers dedicated virtual records (`brand_id=900001` for `Diazepam`, `brand_id=900002` for `Vitamin B-Complex`) backed directly by the real `generics` table metadata in `medicines.db`, coupled with OCR typo-tolerant anchors (`"diazspam"`, `"diazepain"`, `"fibollavin"`, `"pyridaxine"`, `"nicotinamide"`, `"plex table"`).
- **OCR Digit/Unit Repair & Context-Aware Strength Refinement**:
  - `normalize_ocr_for_strength()` strips regulatory license noise (`Mfg. Lic. No.`, `M.A. No.`, `D.A.R. No.`, batch numbers) and repairs common foil OCR confusions (`5OO` -> `500`, `rng`/`mq`/`my`/`m9` -> `mg`, `rncg` -> `mcg`, `rnl` -> `ml`).
  - [`SafetySLMEngine.refine_strength_for_medicine()`](backend/app/services/slm.py) validates extracted strengths against the matched medicine's active ingredient count—weighting lines containing the brand or primary generic word `4×` higher for single-ingredient drugs, and aligning multi-ingredient strengths in exact generic order (e.g., `500 mg + 65 mg` for `Napa Extra: Paracetamol + Caffeine`, and `20 mg + 750 mg + 50 mg + 0.75 mg` for the 4-ingredient `Orthogen` strip in `test/img (1).jpeg`).

#### 3. Similar Generic & Brand Disambiguation (Zero Ghost Matches, 100% True Duplicate Detection)
- **Torn-Foil Substring Ghost Suppression (`SafetySLMEngine.filter_ghost_and_prefix_brands`)**: On torn blister foil, partial generic suffixes like `"dipine"` (from `"Amlodipine"`) or `"epam"` (from `"Diazepam"`) can accidentally fuzzy-match short 4–5 letter brand names in a 17,191-brand database. Our SLM disambiguator checks every candidate single-word brand `b` against all valid dictionary words `w` (`known_db_words`) detected across the image: if `len(w) >= len(b) + 2` and `b in w`, the substring ghost brand is automatically suppressed ([`backend/app/services/slm.py`](backend/app/services/slm.py#L445-L494)).
- **Multi-Word Brand Upgrading (`Napa` + `Extra` -> `Napa Extra`)**: When `"Napa®"` and `"Extra"` (or modifiers in `BRAND_MODIFIERS`: `extra`, `extend`, `plus`, `rapid`, `one`, `max`, `pro`, `forte`, `ds`, `sr`, `xr`, `cr`) appear in separate OCR bounding boxes on the same strip, `filter_ghost_and_prefix_brands()` upgrades the candidate to `"napa extra"` and discards the base prefix `"napa"` so combination strips are never misidentified as single-ingredient plain variants.
- **Length-Ratio & Generic-Exclusion Guards (`MIN_FUZZY_SCORE = 72.0`)**: During brand resolution ([`MedicineMatcher.resolve_brand_from_query()`](backend/app/services/ocr.py#L684-L707)), generic ingredient names are excluded from brand-only anchor pools (`is_generic_ingredient_word()`), and fuzzy candidates must satisfy both `MIN_FUZZY_SCORE = 72.0` (`86.0–90.0` for spatial cluster anchors) and a length-similarity ratio guard (`len_ratio >= 0.65` or `score >= 95.0`) so similar-sounding generics or short noisy tokens never trigger false brand matches.
- **100% Deterministic Multi-Ingredient Decomposition (`evaluate_multi_strip_safety`)**:
  - `_split_active_ingredients()` decomposes every matched medicine's `generic` field across `+`, `/`, `&`, and `,` delimiters into normalized active compounds.
  - If two or more strips share **any** active generic ingredient—whether two plain brands (`Napa 500 mg` + `Ace 500 mg` -> `Paracetamol`) or a plain brand and a combination brand (`Napa 500 mg` + `Napa Extra 500 mg + 65 mg` -> shared `Paracetamol`)—the engine deterministically emits a `high`-severity `duplicate_generic` (`DUPLICATE_ACTIVE_GENERIC_RULE`) alert in both English and Bangla.
  - Non-overlapping active ingredients across all strip pairs are simultaneously cross-checked against `interaction_description` (`DRUG_DRUG_INTERACTION_RULE`) and `drug_class` (`THERAPEUTIC_CLASS_OVERLAP_RULE`), while distinct, non-interacting generics are explicitly verified as safe edges (`PAIRWISE_INGREDIENT_AND_INTERACTION_CHECK`).

#### 4. Non-Oral Administration Route Safety Guardrail (`_enforce_route_safety`)
- A 1B language model or a shared-generic database entry could otherwise suggest oral meal timing (*"খাবারের পরে ভরা পেটে খাবেন"*) for a non-oral product.
- [`_get_non_oral_route_rule()`](backend/app/services/slm.py#L82-L95), [`_enforce_route_safety()`](backend/app/services/slm.py#L97-L134), and [`_extract_form_safe_dosage_snippet()`](backend/app/services/slm.py#L1824-L1870) enforce strict post-generation and database-extraction guardrails for:
  - **Suppositories (`Suppository`)**: *"এটি একটি সাপোজিটরি, যা পায়ুপথে ব্যবহার করতে হয়: মুখে খাওয়া যাবে না।"*
  - **Topical Creams / Ointments / Gels / Lotions / Sprays**: *"এটি বাহ্যিক ব্যবহারের ওষুধ; আক্রান্ত স্থানে পরিষ্কার হাতে পাতলা করে লাগাতে হয়: মুখে খাওয়া যাবে না।"*
  - **Eye / Ear / Nasal Drops (`Drop`, `Ophthalmic`, `Otic`, `Nasal`)**: *"এটি ড্রপ হিসেবে ব্যবহার্য; চিকিৎসকের নির্ধারিত ফোঁটা অনুযায়ী নির্দিষ্ট স্থানে প্রয়োগ করুন: মুখে খাওয়া যাবে না।"*
  - **Injections & Infusions (`Injection`, `Infusion`)**: *"এটি একটি ইনজেকশন, যা শুধুমাত্র চিকিৎসক বা প্রশিক্ষিত নার্সের তত্ত্বাবধানে প্রয়োগ করতে হবে।"*
  - **Inhalers & Nebulisers (`Inhaler`, `Nebuliser`)**: *"এটি শ্বাসের মাধ্যমে (ইনহেলার হিসেবে) গ্রহণ করতে হয়; চিকিৎসকের দেখানো নিয়ম অনুযায়ী ব্যবহার করুন।"*

---

## Architectural Evolution: Proposal (`project.pdf`) vs. Upgraded Production System

Every component proposed in our initial concept document (`project.pdf`) is implemented in production—and we systematically upgraded each layer to achieve **2–3× faster CPU inference**, **higher Bangla clinical accuracy**, **zero cloud cost**, and **100% offline reliability**:

| Architectural Layer | Initial Proposal (`project.pdf`) | Upgraded Production System (`OushudhCheck`) | Why the Production Upgrade Is Faster, More Accurate & Better |
| :--- | :--- | :--- | :--- |
| **1. Quantized Safety SLM & Bangla NLG** | `Qwen2.5-1.5B` or `Llama-3.2-3B` generic instruct model | **Fine-Tuned `Gemma 3 1B-Q4_K_M` GGUF** ([`models/oushudhcheck-gemma3-1b-bn.Q4_K_M.gguf`](models/Modelfile), **768.7 MB**) via `llama.cpp` + **Grounded Bangla Clinical NLG Engine** ([`backend/app/services/slm.py`](backend/app/services/slm.py)) | **2–3× faster CPU latency** (~1.5–2.8s vs. 6–9s on 3B models) and **~800 MB RAM** footprint; fine-tuned specifically on Bangladeshi clinical records from `medicines.db` to keep English brand/strength/generic exact (zero dose transliteration hallucinations) while writing natural Bangla guidance; includes a **30s circuit breaker** with instant deterministic fallback, **100% offline capability**, **unlimited free runs**, and **Bring-Your-Own-Model** via OpenAI-compatible `SLM_API_URL`. |
| **2. Bangladesh Medicine Dataset Engineering** | Raw Kaggle *Assorted Medicine Dataset of Bangladesh* (`21,000+` unjoined rows across 6 CSV files) | **Feature-Engineered & Deduplicated `17,191` Verified Brands + `1,661` Generics + `400+` Drug Classes + `240` Manufacturers** in read-only SQLite + FTS5 ([`data/processed/medicines.db`](data/processed/medicines.db), ~20 MB) + **Virtual EDCL Government Generic Records** | Solves the raw dataset's duplicate rows, missing foreign keys, and HTML/truncation noise (`... Read more`) via [`make_dataset.ipynb`](ml_training/notebooks/make_dataset.ipynb) and [`scripts/ingest_drugs.py`](scripts/ingest_drugs.py); adds **EDCL government strip resolution** (`Diazepam`, `Vitamin B-Complex`) and grounded Bangla `beginner_summary` generation; opens strictly in **read-only URI mode (`?mode=ro`)** for zero database corruption risk. |
| **3. Multi-Strip Vision & Reflective Foil OCR** | Basic `YOLO26n/s` crop -> single-pass `PaddleOCR` | **YOLO26s ONNX 3-Stage Hybrid Spatial Fusion** ([`detector.py`](backend/app/services/detector.py)) + **OpenCV LAB-CLAHE Foil De-Glaring** + **RapidOCR PP-OCRv6 ONNX (`0°` + `90° CCW`)** + **Semantic Ingredient Affinity Clustering** ([`ocr.py`](backend/app/services/ocr.py)) | Eliminates multi-GB PyTorch/PaddlePaddle production container bloat by running pure `onnxruntime` on CPU; **dual-orientation `0° + 90° CCW` OCR** reads vertically printed strips; **Hybrid Spatial Fusion** prevents partial-miss regressions when 5–6 strips touch on a table; **LAB-CLAHE** removes metallic foil glare without distorting text. |
| **4. Telegram Bot & Caregiver Mode** | `aiogram v3` bot wrapper | **Zero-Overhead Async `aiohttp` Long-Polling Engine** ([`bot/telegram_bot.py`](bot/telegram_bot.py)) with **In-Process Service Invocation**, **5-Stage Live Progress Bar (`editMessageText`)**, and **30-Min Stateful `/caregiver` Session Memory** | Invokes `MedicineStripScanner`, `SafetySLMEngine`, and `voice` directly in-process with zero HTTP serialization overhead; streams a real-time **5-stage visual progress bar** (`20% -> 45% -> 70% -> 90% -> Voice MP3`); supports multi-drug text comparisons (`Napa 500mg vs Ace 500mg`) and stateful **Caregiver Mode (`/caregiver`)** voice/text follow-up Q&A. |
| **5. Bangla Voice ASR & Neural TTS** | Basic `faster-whisper` + `edge-tts` | **Dual-Stage Bangla ASR** (`PyAV` 16 kHz mono + `bn-BD` recognizer + offline **`faster-whisper base int8`**) + **`edge-tts` (`bn-BD-NabanitaNeural`)** with **Bangla Numeric-Range Normalizer & Atomic SHA-256 Disk Cache** ([`voice.py`](backend/app/services/voice.py)) | Normalizes clinical ranges like `৩-৪` or `4-6` into natural spoken Bangla (*"৩ থেকে ৪"*) instead of speaking *"minus"*; filters non-Bangla script hallucinations; caches synthesized MP3 clips on disk (200 files / 100 MB LRU) for **instant `<15ms` replay** on repeated queries. |
| **6. Web Application & Judge Observability** | Next.js 16 + Tailwind CSS scan page | **Next.js 16.4.6 + React 19.3 + TypeScript 5 + Tailwind CSS 4** with **NDJSON Live Stage Streaming (`/scan-stream`)**, **6-Stage AI Transparency Inspector (`/pipeline`)**, **Pairwise Safety-Edge Graph**, and **Live Detector Mode Controls** | Gives judges and clinicians complete glass-box visibility into per-stage latencies (`ms`), raw vs. CLAHE-enhanced strip thumbnails, YOLO vs. OCR confidence scores, per-strip provenance (`yolo26_direct` vs. `ocr_fallback`), and live query overrides (`detector_mode=hybrid_fusion\|yolo_only\|ocr_only`, `yolo_conf`). |

---

## System Architecture & End-to-End Data Flow

```mermaid
flowchart TB
    subgraph Clients["User & Caregiver Interfaces"]
        Web["Next.js 16.4.6 Web App (:3000)\n• /scan (Multi-Strip Scanner & Voice Player)\n• /database (17,191 BD Medicine Directory)\n• /pipeline (6-Stage AI Transparency Inspector)"]
        TGBot["Telegram Bot @OushudhCheck_bot\n(bot/telegram_bot.py)\n• 1–6 Strip Photo Scan (5-Stage Progress)\n• /caregiver Voice & Text Follow-Up Q&A (30m TTL)\n• Multi-Medicine Text Comparison"]
    end

    subgraph Core["OushudhCheck Python 3.12 Inference Core (100% Local CPU)"]
        API["FastAPI Backend (:8000)\n(backend/app/main.py & api/routes.py)\n• Non-blocking threadpool execution\n• NDJSON real-time stage streaming (/scan-stream)"]
        Scanner["MedicineStripScanner\n(backend/app/services/ocr.py)\nUnified 5-Stage Hybrid Runner"]
        Detector["StripDetector\n(backend/app/services/detector.py)\n• YOLO26s ONNX (CPUExecutionProvider)\n• OpenCV LAB-CLAHE + Unsharp Mask"]
        OCR["OCREngine & MedicineMatcher\n(backend/app/services/ocr.py)\n• RapidOCR PP-OCRv6 ONNX (0° + 90° CCW)\n• RapidFuzz WRatio + EDCL Virtual Records\n• Deterministic Duplicate & DDI Graph"]
        SLM["SafetySLMEngine & Caregiver Q&A\n(backend/app/services/slm.py)\n• Ghost-brand suppression & strength alignment\n• 9-intent Caregiver Q&A + Route Safety Guardrail\n• 30s circuit breaker + deterministic Bangla NLG"]
        Voice["Bangla Voice Engine\n(backend/app/services/voice.py)\n• PyAV 16kHz + bn-BD ASR + faster-whisper (base int8)\n• Edge Neural TTS (bn-BD-NabanitaNeural) + SHA-256 Cache"]
    end

    subgraph Storage["Local Read-Only Data & Quantized Model Artifacts"]
        DB[("data/processed/medicines.db\nSQLite Read-Only (?mode=ro) + FTS5\n17,191 Brands | 1,661 Generics | 400+ Classes")]
        ONNX[("data/processed/yolo_strips.onnx\nFine-Tuned YOLO26s Blister-Strip Detector\nStatic FP32 ONNX (opset 17, 640x640)")]
        GGUF[["llama-server (:8081) / Ollama\nmodels/oushudhcheck-gemma3-1b-bn.Q4_K_M.gguf\nFine-Tuned Bangla Clinical Gemma 3 1B (768.7 MB)"]]
    end

    Web -->|"REST & NDJSON Stream\n(/api/v1/scan-stream, /medicines, /tts)"| API
    API --> Scanner
    API --> Voice
    TGBot -->|"Direct In-Process Python Import"| Scanner
    TGBot -->|"Direct In-Process Python Import"| SLM
    TGBot -->|"Direct In-Process Python Import"| Voice
    Scanner --> Detector
    Scanner --> OCR
    Scanner --> SLM
    Detector --> ONNX
    OCR --> DB
    SLM -->|"OpenAI-Compatible /v1/chat/completions\n(with instant deterministic fallback)"| GGUF
```

---

## Why OushudhCheck Matters: Cost, Privacy, Offline Autonomy & Real-World Use Cases in Bangladesh

### 1. Built 100% for Bangladesh's Pharmaceutical Ecosystem
Unlike Western pill-identifier apps that rely on imprinted pill codes or US FDA NDC barcodes (which do not exist on Bangladeshi loose blister strips), OushudhCheck is purpose-built for **Bangladesh's pharmaceutical landscape**:
- Covers **17,191 brands** and **1,661 generics** across **240+ Bangladeshi manufacturers**, including **Square Pharmaceuticals**, **Beximco Pharmaceuticals**, **Incepta Pharmaceuticals**, **Renata**, **Eskayef (SK+F)**, **Aristopharma**, **ACME Laboratories**, **Healthcare Pharmaceuticals**, **Opsonin Pharma**, **ACI Limited**, **Popular Pharmaceuticals**, **Radiant**, **Ziska**, **Beacon**, **Ibn Sina**, **Orion**, **Drug International**, and **Essential Drugs Co. Ltd. (EDCL)** government hospital supplies.
- Specifically engineered for **silver/white metallic aluminum back-foil**, **Bangladeshi regulatory license formats** (`Mfg. Lic. No.`, `D.A.R. No.`, `M.A. No.`), and **everyday Bangla clinical vocabulary** (`খালি পেটে`, `ভরা পেটে`, `সাপোজিটরি`, `স্বাস্থ্য বাতায়ন ১৬২৬৩`).

### 2. Zero API Cost ($0.00/Scan), Unlimited Runs & Bring-Your-Own-Model Freedom
- **No Paid Cloud LLM Dependencies**: Every AI component—`YOLO26s ONNX`, `RapidOCR PP-OCRv6 ONNX`, `RapidFuzz + SQLite FTS5`, `Gemma 3 1B-Q4_K_M GGUF` (`llama.cpp`), and `faster-whisper (base, int8)`—runs locally on standard CPU hardware.
- **Unlimited Scans with Zero Rate Limits**: Clinics, pharmacies, and households can run thousands of scans per day without API keys, per-token charges, or quota exhaustion.
- **Bring-Your-Own-Model (BYOM) OpenAI-Compatible Architecture**: Because [`backend/app/services/slm.py`](backend/app/services/slm.py) communicates via standard `/v1/chat/completions` (`SLM_API_URL` and `SLM_MODEL_NAME`), hospitals or developers can swap in any custom GGUF model via `llama-server`, `Ollama`, `vLLM`, or LM Studio by changing a single environment variable in [`.env`](.env.example)—and even if the SLM server is stopped, the deterministic grounded Bangla NLG engine continues serving verified clinical summaries and caregiver Q&A with **zero downtime**.

### 3. 100% Medical Privacy & Read-Only Data Integrity
- **On-Device / On-Premise Privacy**: Sensitive patient medicine photos, voice questions, and chronic-care medication lists never leave the local machine or hospital server for third-party cloud LLM processing.
- **Immutable Pharmaceutical Database**: Every connection to [`data/processed/medicines.db`](data/processed/medicines.db) uses SQLite URI read-only mode (`?mode=ro` in [`backend/app/core/config.py`](backend/app/core/config.py#L62-L74)), guaranteeing that runtime queries can never mutate or corrupt verified clinical records.

### 4. High-Impact Real-World Use Cases Across Bangladesh

| Target Community | Real-World Scenario in Bangladesh | How OushudhCheck Delivers Immediate Impact |
| :--- | :--- | :--- |
| **Rural Households & Low-Connectivity Areas** | Families keep loose, half-cut blister strips in a box at home with no internet access for cloud AI and limited English literacy. | Runs **100% offline on local CPU** (or via lightweight Telegram messages on 2G/3G); identifies 1–6 strips in one photo and speaks clear Bangla dosage and meal-timing rules out loud. |
| **Elderly Patients & Family Caregivers (`/caregiver`)** | An elderly parent takes hypertension (`Amdocal`), gastric (`Seclo`), and pain (`Napa Extra` / `Tufnil`) strips daily; a family caregiver wants to check safety and ask follow-up questions in Bangla. | **Caregiver Mode (`/caregiver`)** remembers the scanned strips for 30 minutes and answers spoken Bangla voice questions (*"সকালে খালি পেটে খাবে নাকি ভরা পেটে?"*, *"পার্শ্বপ্রতিক্রিয়া কী?"*) with spoken Bangla audio replies. |
| **NGO Health Camps, Upazila Health Workers & Community Clinics** | Frontline health workers (`স্বাস্থ্য বাতায়ন` / community clinic staff) review bags of loose strips brought by chronic-care patients during high-volume rural health camps. | Screens up to 6 strips simultaneously in under 5 seconds on a single laptop or phone, instantly flagging duplicate generics (e.g., two different brands of Paracetamol or Omeprazole) including **EDCL government strips**. |
| **Busy Local Community Pharmacies** | Neighborhood pharmacy dispensers serve dozens of customers per hour at busy counters where brand substitutions are common. | Provides a fast counter-verification aid via the Web Scanner (`/scan`) or multi-drug text search (`Napa 500mg vs Ace 500mg`), catching accidental duplicate-generic dispensing and DDI conflicts before medicines leave the counter. |

---

## Complete Model Training & Dataset Engineering Pipeline (`ml_training/` & `scripts/`)

Every dataset and model in OushudhCheck is 100% reproducible from our open training notebooks in [`ml_training/notebooks/`](ml_training/notebooks/) and data ingestion scripts in [`scripts/`](scripts/).

```mermaid
flowchart LR
    Raw["Kaggle Assorted Medicine Dataset of BD\n(6 CSVs: 21,714 raw brand rows)"] --> NB1["1. ml_training/notebooks/make_dataset.ipynb\n Relational Join, Key Normalization & Cleaning"]
    NB1 --> CleanCSV["data/raw/master_medicine_bd_clean.csv"]
    CleanCSV --> Ingest["2. scripts/ingest_drugs.py\nDeduplication + Normalization + SQLite FTS5"]
    Ingest --> SQLite[("data/processed/medicines.db\n17,191 Brands | 1,661 Generics")]
    SQLite --> NB3["3. ml_training/notebooks/train-model-oshudcheck.ipynb\nBalanced SFT Generation + Unsloth 2xT4 DDP LoRA\n+ Q4_K_M GGUF Export"]
    NB3 --> GGUF[("models/oushudhcheck-gemma3-1b-bn.Q4_K_M.gguf\n(768.7 MB) + Modelfile + system_prompt.txt")]
    PubDS["Public Blister Datasets +\n3,000 Procedural BD Foil Tabletop Scenes"] --> NB2["4. ml_training/notebooks/train_yolo26s_blister_strips.ipynb\nUltralytics YOLO26s Training + Static FP32 ONNX Export"]
    NB2 --> YOLO[("data/processed/yolo_strips.onnx\nVerified Pure ONNXRuntime CPU Parity")]
```

### 1. Raw Dataset Consolidation & Cleaning ([`ml_training/notebooks/make_dataset.ipynb`](ml_training/notebooks/make_dataset.ipynb))
- **Source Data**: Downloads the 6 fragmented CSV tables of the [Assorted Medicine Dataset of Bangladesh](https://www.kaggle.com/datasets/ahmedshahriarsakib/assorted-medicine-dataset-of-bangladesh) via `kagglehub`:
  - `medicine.csv` (`21,714 rows × 10 cols`)
  - `generic.csv` (`1,711 rows × 22 cols`)
  - `manufacturer.csv` (`240 rows × 5 cols`)
  - `dosage form.csv` (`113 rows × 4 cols`)
  - `drug class.csv` (`453 rows × 4 cols`)
  - `indication.csv` (`2,043 rows × 4 cols`)
- **Feature Engineering & Cleaning**: Normalizes join keys (`_generic_key`, `_manufacturer_key`, `_dosage_key`, `_class_key`, `_indication_key`), deduplicates lookup tables, performs left relational joins across all 6 tables into a unified clinical schema, strips HTML entities and scraped truncation artifacts, and exports `data/raw/master_medicine_bd_clean.csv`.

### 2. SQLite & FTS5 Database Ingestion ([`scripts/ingest_drugs.py`](scripts/ingest_drugs.py))
- **Deduplication (`21,714` raw -> `17,191` verified brands & `1,661` unique generics)**:
  - Deduplicates brand records on `(brand_lower, generic_lower, dosage_form_lower, manufacturer_lower)` while backfilling any missing generic clinical fields (`drug_class`, `indication`, `dosage_description`, `interaction_description`, `contraindications_description`, `side_effects_description`, `pregnancy_and_lactation_description`, `precautions_description`, `storage_conditions_description`) across rows sharing the same generic compound.
  - Strips scraped `... Read more` prefixes via `_READ_MORE_RE` and normalizes null tokens (`nan`, `NULL`, `N/A`, `-`).
- **Database Schema & Full-Text Search Index**:
  - Creates normalized `brands` and `generics` tables with B-tree indexes (`idx_brands_id`, `idx_brands_lower`, `idx_brands_generic`, `idx_generics_lower`).
  - Creates the unified `medicines` SQL view and the `medicines_fts` **SQLite FTS5** virtual table (`tokenize='unicode61'`), then runs `VACUUM` to produce the self-contained ~20 MB [`data/processed/medicines.db`](data/processed/medicines.db).
- **Reproduce Locally**:
  ```bash
  python scripts/ingest_drugs.py
  ```

### 3. YOLO26s Blister-Strip Detector Training & ONNX Export ([`ml_training/notebooks/train_yolo26s_blister_strips.ipynb`](ml_training/notebooks/train_yolo26s_blister_strips.ipynb))
- **Domain-Bridge Dataset Pipeline**:
  1. **Public Whole-Strip Sanitization**: Scans public blister-pack YOLO datasets, converts polygon/bbox annotations into single-class `0: medicine_strip`, and filters out individual pill-cavity annotations whose median box area is `< 1.5%` of the image (`MIN_MEDIAN_BOX_AREA = 0.015`) with MD5 deduplication and an `85/15` train/val split.
  2. **Procedural Bangladeshi Metallic Back-Foil Tabletop Synthesizer**: Because public datasets skew toward front plastic bubbles, Cell 3 procedurally generates **~3,000 realistic Bangladeshi multi-strip tabletop scenes** (`2,600` train + `350` val, including `10%` empty-tabletop hard negatives) rendering 1–6 silver/white/tinted metallic foil strips printed with real Bangladeshi brand/generic typography (`Napa Extra`, `Seclo 20`, `Amdocal 5`, `Tufnil 200`, `Diazepam 5`, `Vitamin B-Complex`, etc.), embossed pill cavities, specular glare streaks, torn half-used polygon masks (`35%` probability), `0°–360°` rotations, feathered drop shadows, and camera blur/JPEG compression.
- **Training & Pure ONNXRuntime Parity Verification**:
  - Trains `yolo26s.pt` on Kaggle 2× T4 GPUs via Ultralytics CLI (`imgsz=640`, `epochs=50`, `patience=12`, `batch=32`, `amp=True`, `cos_lr=True`, `mosaic=1.0`, `close_mosaic=10`, `degrees=15.0`, `flipud=0.5`, `fliplr=0.5`).
  - Exports static FP32 ONNX (`format="onnx"`, `imgsz=640`, `opset=17`, `simplify=True`, `dynamic=False`, `half=False`) with native end-to-end NMS-free output shape `(1, 300, 6)` (`[x1, y1, x2, y2, conf, class_id]`), validates the graph with `onnx.checker.check_model`, and verifies box-count parity between PyTorch `.pt` and pure `onnxruntime` (`CPUExecutionProvider`) before saving to [`data/processed/yolo_strips.onnx`](data/processed/yolo_strips.onnx).

### 4. Gemma 3 1B Bangla Clinical Fine-Tuning & Q4_K_M GGUF Export ([`ml_training/notebooks/train-model-oshudcheck.ipynb`](ml_training/notebooks/train-model-oshudcheck.ipynb))
- **Grounded Bangla SFT Dataset Construction (`2,671` Train / `140` Eval Examples)**:
  - Connects directly to `medicines.db` (`17,191` brands / `1,661` generics) and performs balanced sampling of up to 2 brands per unique generic across all 1,661 generics.
  - **Clinical Hybrid Language Rule**: Keeps **medicine brand names, numeric dosage strengths (`500 mg + 65 mg`, `20 mg`), and generic ingredient names in exact English** so numbers and pharmaceutical compounds are never corrupted by transliteration, while generating all explanations, dosage forms (`ট্যাবলেট`, `ক্যাপসুল`, `সিরাপ`, `সাপোজিটরি`), meal timings, side effects, and precautions in fluent, natural Bangla.
  - Respects negation (`NO_SIDE_EFFECT_RE`: never fabricates side effects when `medicines.db` states *"no known side effects"* or *"generally well tolerated"*) and enforces non-oral administration routes for creams, eye drops, inhalers, and injections.
- **Multi-GPU DDP QLoRA Fine-Tuning (Kaggle 2× Tesla T4)**:
  - Base Model: `unsloth/gemma-3-1b-it-unsloth-bnb-4bit` (`max_seq_length = 1024`).
  - LoRA Configuration: `r = 16`, `lora_alpha = 32`, `lora_dropout = 0.05`, targeting language, attention, and MLP modules (`13,045,760` trainable parameters = `1.93%` of `675,994,752`).
  - Training Recipe: `SFTTrainer` with `train_on_responses_only` (`<start_of_turn>model\n`), `packing = False` (so safety warnings never bleed across medicine records), `per_device_train_batch_size = 4`, `gradient_accumulation_steps = 2` across 2 GPUs (effective batch size `16`), `learning_rate = 2e-4` (cosine schedule), `2 epochs` (`334 steps`), converging to **`train_loss = 0.2297`** and **`eval_loss = 0.07006`** in `21m 23s`.
- **Faithfulness Audit & `Q4_K_M` GGUF Quantization**:
  - Audits the trained adapter on 5 real Bangladeshi benchmark drugs (`Napa Extra 500 mg + 65 mg`, `Seclo 20 mg`, `Amdocal 5 mg`, `Tufnil 200 mg`, `Diazepam 5 mg`) verifying 100% preservation of exact English brand/strength/generic tokens and fluent Bangla guidance.
  - Merges LoRA adapters to 16-bit and quantizes via `llama.cpp` to `Q4_K_M` GGUF: **`oushudhcheck-gemma3-1b-bn.Q4_K_M.gguf` (768.7 MB)**, packaged alongside [`models/Modelfile`](models/Modelfile) and [`models/system_prompt.txt`](models/system_prompt.txt) in `oushudhcheck_gemma3_1b_deploy.zip` on GitHub Releases.

---

## Complete REST API Reference

The FastAPI backend ([`backend/app/main.py`](backend/app/main.py) and [`backend/app/api/routes.py`](backend/app/api/routes.py)) mounts all REST routes at both `/api/v1` and `/api`, with interactive Swagger UI at `http://localhost:8000/docs`.

| Method & Endpoint | Query / Body Parameters | Response Schema / Content | Purpose & Key Details |
| :--- | :--- | :--- | :--- |
| **`GET /health`** | *(none)* | `{ status, service, detector }` | Returns service health plus full `StripDetector.get_model_info()` diagnostics (ONNX filename, size in MB, SHA-256 prefix, `isRealModel`, input/output tensor shapes, active ORT threads, and pipeline mode). |
| **`GET /api/v1/medicines`** | `q` (str), `dosageForm` (str, default `"All"`), `page` (int `>=1`), `limit` (int `1..100`, default `24`) | `PaginatedMedicinesResponse` | Queries `medicines.db` in read-only mode (`?mode=ro`) across brand, generic, manufacturer, drug class, and indication; strips strength numbers from `q`; generates grounded Bangla `beginner_summary` for every item and returns `{ totalBrands: 17191, totalGenerics: 1661 }`. |
| **`GET /api/v1/medicines/search`** | `q` (str, `min_length=2`), `limit` (int `1..50`, default `10`) | `List[MedicineInfo]` | Fast prefix-prioritized brand and generic lookup returning full clinical metadata and Bangla `beginner_summary`. |
| **`POST /api/v1/scan`** | `file` (`UploadFile`, max 15 MB / 25 MP), `detector_mode` (`hybrid_fusion` \| `yolo_only` \| `ocr_only`), `yolo_conf` (`0.05..0.95`) | Full Scan Payload (`scanId`, `detectedStrips`, `alerts`, `banglaVoiceScript`, `audioUrl`, `transparencyInspector`) | Executes the complete 6-stage AI pipeline on an uploaded photo, returning per-strip bounding boxes, cropped & CLAHE-preprocessed base64 thumbnails, pairwise safety edges, Bangla summaries, and TTS audio URL. |
| **`POST /api/v1/scan-stream`** | `file` (`UploadFile`), `detector_mode` (optional), `yolo_conf` (optional) | `application/x-ndjson` stream | Streams real-time stage progress events (`{"type": "stage", "stageIndex": 0..4, "status": "running"\|"done", "durationMs": ...}`) as each AI stage completes, followed by `{"type": "result", "data": ...}`. |
| **`POST /api/v1/scan-benchmark`** | `sample_id` (str, default `"multi-5-strips"`) | Full Scan Payload + `sampleId`, `sampleTitle`, `demoImageUrl` | Runs the live 6-stage AI pipeline on a real benchmark image from [`test/`](test/) so judges can inspect live execution timings and cropped thumbnails on demand. |
| **`POST /api/v1/extract-text`** | JSON `TextExtractRequest`: `{"text": "..."}` | `StripScanResult` | Runs `MedicineMatcher.match_strip_text()` + `SafetySLMEngine.refine_strength_for_medicine()` on raw OCR text from a single strip. |
| **`POST /api/v1/extract-multi-text`** | JSON `MultiTextExtractRequest`: `{"strip_texts": ["...", "..."]}` | `MultiStripScanResponse` | Resolves multiple raw OCR strip texts and runs `evaluate_multi_strip_safety()` for duplicate active generics and DDI alerts. |
| **`GET /api/v1/tts`** | `text` (str, `1..3500` chars) | `audio/mpeg` (`.mp3` `FileResponse`) | Synthesizes natural Bangla speech via `edge-tts` (`bn-BD-NabanitaNeural`, rate `-5%`) with numeric-range normalization and immutable cache headers. |
| **`POST /api/v1/tts`** | JSON `TTSRequest`: `{"text": "..."}` (`1..6000` chars) | `audio/mpeg` (`.mp3` `FileResponse`) | Synthesizes long multi-strip Bangla summaries up to 6,000 characters via JSON POST body without URL length limits. |
| **`GET /api/v1/tts/clip/{audio_id}`** | Path param `audio_id` (16-hex-char SHA-256 key) | `audio/mpeg` (`.mp3` `FileResponse`) | Serves a pre-synthesized cached Bangla MP3 clip from the atomic SHA-256 disk cache (`AUDIO_CACHE_DIR`). |
| **`GET /api/v1/benchmark-samples`** | *(none)* | `{"samples": [...]}` | Lists the 6 real Bangladeshi tabletop benchmark samples available in [`test/`](test/) for the `/pipeline` AI Transparency Inspector. |
| **`GET /api/v1/demo-image/{sample_id}`** | Path param `sample_id` | `image/jpeg` or `image/png` | Serves the original high-resolution benchmark image from [`test/`](test/) for visual inspection and bounding-box overlay rendering. |

---

## Test & Benchmark Suite (`test/` & `tests/`)

Both [`test/`](test/) (real Bangladeshi medicine strip benchmark photos and caregiver voice clips) and [`tests/`](tests/) (automated pytest verification suites) are tracked directly in the repository (verified not gitignored) so human judges and automated AI grading agents can run every benchmark and test out of the box.

### 1. Real Bangladeshi Medicine Strip Benchmark Photos ([`test/`](test/))

| Image File | Benchmark Sample ID | Strips in Photo | Real Bangladeshi Medicine Strips & Clinical Challenge Tested |
| :--- | :--- | :---: | :--- |
| [`test/newimg (3).jpg`](test/newimg%20(3).jpg) | `multi-5-strips` | **5** | **5-Strip Crowded Tabletop**: `Vitamin B-Complex` (EDCL), `Napa Extra` (`500 mg + 65 mg`), `Diazepam` (`5 mg`, EDCL), `Amdocal` (`5 mg`), `Tufnil` (`200 mg`). Tests multi-strip spatial clustering, split `"Napa" + "Extra"` upgrade, substring ghost suppression (`"dipine"`, `"epam"`), and EDCL government generic strips. |
| [`test/newimg (2).jpg`](test/newimg%20(2).jpg) | `multi-4-strips` | **4** | **4-Strip Tabletop**: `Diazepam` (`5 mg`, EDCL), `Napa Extra` (`500 mg + 65 mg`), `Tufnil` (`200 mg`), `Amdocal` (`5 mg`). Tests 4-strip orientation and multi-ingredient strength pairing. |
| [`test/newimg (1).jpg`](test/newimg%20(1).jpg) | *(additional test photo)* | **3** | **3-Strip Tabletop**: `Diazepam` (`5 mg`, EDCL), `Vitamin B-Complex` (EDCL), `Napa Extra` (`500 mg + 65 mg`). |
| [`test/img1.jpg`](test/img1.jpg) | `multi-2-emistat-fluver` | **2** | **2-Strip Reflective Metallic Foil**: `Emistat` (`8 mg`, Ondansetron) & `Fluver` (`10 mg`, Flunarizine). Tests specular metallic foil glare removal and bare-number strength recovery. |
| [`test/img (4).jpeg`](test/img%20(4).jpeg) | `multi-2-seronex-fluver` | **2** | **2-Strip Glass Tabletop**: `Seronex` (`50 mg`, Sertraline) & `Fluver` (`10 mg`, Flunarizine). Tests high-glare glass surface reflection. |
| [`test/img (1).jpeg`](test/img%20(1).jpeg) | `single-orthogen-combo` | **1** | **4-Ingredient Combination Strip**: `Orthogen` (`20 mg + 750 mg + 50 mg + 0.75 mg`). Tests 4-compound ordered strength assembly (`SafetySLMEngine.refine_strength_for_medicine`). |
| [`test/img (2).jpeg`](test/img%20(2).jpeg) | `single-naprosyn-vertical` | **1** | **Vertical 90° Printed Strip**: `Naprosyn` (`500 mg`, Naproxen). Tests dual-orientation RapidOCR (`90° CCW` pass). |
| [`test/img (3).jpeg`](test/img%20(3).jpeg) | *(additional test photo)* | **2** | **2-Strip Neuropathic & Anticonvulsant**: `Prelica` (`50 mg`, Pregabalin) & `Epitra` (`1 mg`, Clonazepam). |
| [`test/img (5).jpeg`](test/img%20(5).jpeg) | *(additional test photo)* | **2** | **2-Strip Foil Variant**: `Prelica` (`50 mg`, Pregabalin) & `Epitra` (`1 mg`, Clonazepam) under angled lighting. |

#### Run the Live Image Benchmark Suite ([`scripts/benchmark.py`](scripts/benchmark.py))
```bash
python scripts/benchmark.py
```

### 2. Caregiver Mode Voice Examples ([`test/caregiver_voice_examples/`](test/caregiver_voice_examples/README.txt))
- [`test/caregiver_voice_examples/01_meal_timing.mp3`](test/caregiver_voice_examples/01_meal_timing.mp3)
- [`test/caregiver_voice_examples/02_side_effects.mp3`](test/caregiver_voice_examples/02_side_effects.mp3)
- [`test/caregiver_voice_examples/03_pregnancy_safety.mp3`](test/caregiver_voice_examples/03_pregnancy_safety.mp3)
- [`test/caregiver_voice_examples/04_interaction_warning.mp3`](test/caregiver_voice_examples/04_interaction_warning.mp3)
- [`test/caregiver_voice_examples/README.txt`](test/caregiver_voice_examples/README.txt)

### 3. Automated Verification Test Suite ([`tests/test_caregiver_qa.py`](tests/test_caregiver_qa.py))
Run the automated test suite verifying **Onsite Challenge 1** (Caregiver Mode 9-intent Bangla Q&A, zero English prose leaks, zero boilerplate regurgitation, morning/bedtime contradiction guards, multi-strip focus, non-oral route safety, and Telegram session memory) and **Onsite Challenge 2** (substring ghost suppression, multi-word brand upgrades, deterministic duplicate active generic detection, and strength refinement):

```bash
python -m pytest tests/ -v
```

---

## How to Run, Deploy & Configure Custom GGUF Models

### Option 1: One-Command Docker Compose (Recommended Full Stack)

Run the entire 5-service stack (**`model-init` auto-downloader** + **`llama.cpp` Gemma 3 1B server** + **FastAPI backend** + **Next.js 16 frontend** + **Telegram Bot**) with a single command:

```bash
cp .env.example .env
# (Optional) Set TELEGRAM_BOT_TOKEN in .env to enable @OushudhCheck_bot
docker compose up --build
```

- **Automatic Model Bootstrap (`model-init`)**: On first startup, the `model-init` container executes [`scripts/download_model.sh`](scripts/download_model.sh) to download `oushudhcheck_gemma3_1b_deploy.zip` from [GitHub Releases](https://github.com/rhridoyhossan/OushudhCheck/releases/latest) into the persistent `oushudhcheck_models` Docker volume and extracts `oushudhcheck-gemma3-1b-bn.Q4_K_M.gguf` (`768.7 MB`), `system_prompt.txt`, and `Modelfile`. Subsequent restarts skip the download automatically.
- **Exposed Endpoints**:
  - **Next.js 16 Web Application**: `http://localhost:3000` (`/scan`, `/database`, `/pipeline`)
  - **FastAPI Backend & Interactive Swagger UI**: `http://localhost:8000/docs`
  - **Backend Health & YOLO26 Diagnostics**: `http://localhost:8000/health`
  - **Gemma 3 1B `llama-server` API**: `http://localhost:8081/health`

### Option 2: Run Locally Without Docker (Windows / Linux / macOS)

#### Step 1: Clone & Configure Environment
```bash
cp .env.example .env
```

#### Step 2: Download the Fine-Tuned Gemma 3 1B GGUF Release Model
- **Automatic script (Git Bash / Linux / macOS / WSL):**
  ```bash
  MODEL_DIR=./models sh scripts/download_model.sh
  ```
- **Manual download:**
  Download [`oushudhcheck_gemma3_1b_deploy.zip`](https://github.com/rhridoyhossan/OushudhCheck/releases/latest/download/oushudhcheck_gemma3_1b_deploy.zip) from the latest GitHub Release and extract it into `models/` so that `models/oushudhcheck-gemma3-1b-bn.Q4_K_M.gguf`, [`models/system_prompt.txt`](models/system_prompt.txt), and [`models/Modelfile`](models/Modelfile) are present.
  *(Note: Even if `llama-server` is not running locally, the backend and Telegram bot automatically use the built-in deterministic grounded Bangla clinical engine with zero errors.)*

#### Step 3A: One-Click Windows Full-Stack Launcher
On Windows, simply double-click or run [`run_local.bat`](run_local.bat) from the repository root:
```bat
run_local.bat
```
This script verifies `.venv`, installs dependencies, and launches `llama-server` (`:8081`), FastAPI (`:8000`), the Telegram Bot (`@OushudhCheck_bot`), and the Next.js Frontend (`:3000`) in separate terminal windows.

#### Step 3B: Manual Terminal Startup (Windows PowerShell / Linux / macOS)

1. **Create virtual environment & install Python dependencies:**
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # Linux / macOS / Git Bash:
   # source .venv/bin/activate

   pip install -r backend/requirements.txt -r bot/requirements.txt
   ```

2. **Start the Gemma 3 1B Bangla SLM (`llama-server` on port `8081`):**
   ```bash
   llama-server -m models/oushudhcheck-gemma3-1b-bn.Q4_K_M.gguf --host 127.0.0.1 --port 8081 -c 8192 -np 2 -fa on --cache-reuse 512
   ```

3. **Start the FastAPI Backend (`http://localhost:8000`):**
   ```bash
   python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Start the Next.js 16 Frontend (`http://localhost:3000`):**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

5. **Start the Telegram Bot (`@OushudhCheck_bot`):**
   ```bash
   python bot/telegram_bot.py
   ```

### Option 3: Bring-Your-Own-Model (Ollama / Custom GGUF / OpenAI-Compatible Server)

OushudhCheck supports any custom GGUF model or OpenAI-compatible inference server out of the box:

- **Using Ollama with our included [`models/Modelfile`](models/Modelfile):**
  ```bash
  cd models
  ollama create oushudhcheck-gemma3-1b -f Modelfile
  ```
  Then point `.env` to Ollama's OpenAI-compatible endpoint:
  ```env
  SLM_MODEL_NAME=oushudhcheck-gemma3-1b
  SLM_API_URL=http://127.0.0.1:11434/v1/chat/completions
  ```
- **Using any custom `.gguf` file in Docker or `llama-server`:**
  Place your `.gguf` file in `models/` (or set `MODEL_RELEASE_URL` / `MODEL_FILENAME` in `.env` to your own `.gguf` or `.zip` release URL) and update `SLM_GGUF_PATH` / `SLM_MODEL_NAME` in [`.env`](.env.example).

### Environment Configuration Reference ([`.env.example`](.env.example))

| Environment Variable | Default Value | Description |
| :--- | :--- | :--- |
| `TELEGRAM_BOT_TOKEN` | `your_telegram_bot_token_here` | Telegram Bot API token from `@BotFather` for `@OushudhCheck_bot`. |
| `TELEGRAM_BOT_USERNAME` | `OushudhCheck_bot` | Telegram bot handle displayed in `/start` and help commands. |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000/api/v1` | Browser-accessible FastAPI base URL for the Next.js frontend. |
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated CORS origins allowed by FastAPI. |
| `MEDICINES_DB_PATH` | `data/processed/medicines.db` | Path to the read-only SQLite Bangladesh medicine database. |
| `YOLO_MODEL_PATH` | `data/processed/yolo_strips.onnx` | Path to the trained YOLO26s ONNX blister-strip detector weights. |
| `YOLO_PIPELINE_MODE` | `hybrid_fusion` | Active vision fusion mode: `hybrid_fusion` \| `yolo_only` \| `ocr_only`. |
| `YOLO_CONF_THRESHOLD` | `0.22` | Minimum YOLO26s box confidence threshold (`0.05..0.95`). |
| `MIN_FUZZY_SCORE` | `72.0` | Minimum RapidFuzz `WRatio` score for brand matching. |
| `SLM_MODEL_NAME` | `oushudhcheck-gemma3-1b` | Model identifier sent in `/v1/chat/completions` requests. |
| `SLM_API_URL` | `http://127.0.0.1:8081/v1/chat/completions` | OpenAI-compatible chat completions endpoint (`llama-server`, Ollama, etc.). |
| `SLM_GGUF_PATH` | `models/oushudhcheck-gemma3-1b-bn.Q4_K_M.gguf` | Local filesystem path to the quantized Q4_K_M GGUF weights. |
| `SLM_MAX_TOKENS` | `768` | Maximum tokens generated per Bangla clinical summary request. |
| `SLM_TIMEOUT_SEC` | `30.0` | Timeout before tripping the 30s circuit breaker to grounded fallback mode. |
| `MODEL_RELEASE_URL` | `https://github.com/rhridoyhossan/OushudhCheck/releases/latest/download/oushudhcheck_gemma3_1b_deploy.zip` | GitHub Release archive auto-downloaded by [`scripts/download_model.sh`](scripts/download_model.sh). |

---

## Repository Structure

```text
OushudhCheck/
├── backend/
│   ├── app/
│   │   ├── api/routes.py             # All 12 FastAPI REST & NDJSON streaming endpoints
│   │   ├── core/config.py            # Path resolution, thresholds & read-only SQLite (?mode=ro)
│   │   ├── models/schemas.py         # Pydantic v2 request/response contracts
│   │   ├── services/
│   │   │   ├── detector.py           # StripDetector (YOLO26s ONNX + LAB-CLAHE foil preprocessor)
│   │   │   ├── ocr.py                # OCREngine (PP-OCRv6 0°+90°), MedicineMatcher, Hybrid Fusion & Safety Graph
│   │   │   ├── slm.py                # SafetySLMEngine, Gemma 3 1B client, Caregiver Q&A & Route Safety Guardrails
│   │   │   └── voice.py              # Bangla ASR (bn-BD + faster-whisper int8) & Edge Neural TTS + SHA-256 Cache
│   │   └── main.py                   # FastAPI application entrypoint & /health diagnostics
│   └── requirements.txt              # Backend Python dependencies
├── bot/
│   ├── telegram_bot.py               # Async aiohttp Telegram bot (@OushudhCheck_bot) with /caregiver mode
│   └── requirements.txt              # Telegram bot dependencies
├── frontend/
│   ├── app/
│   │   ├── page.tsx                  # Landing page & healthcare showcase
│   │   ├── scan/page.tsx             # Interactive Multi-Strip Medicine Scanner workspace
│   │   ├── database/page.tsx         # Searchable 17,191 BD Medicine Directory
│   │   ├── pipeline/page.tsx         # Live 6-Stage AI Pipeline & Transparency Inspector
│   │   └── api/medicines/route.ts    # Next.js API route handler with local SQLite read-only fallback
│   ├── components/                   # Scanner workspace, AI Transparency Inspector & UI components
│   ├── lib/                          # Typed FastAPI client (api.ts), interfaces (types.ts) & utilities
│   └── Dockerfile                    # Multi-stage Node 20-alpine standalone production container
├── data/
│   └── processed/
│       ├── medicines.db              # Curated read-only SQLite DB (17,191 BD brands / 1,661 generics + FTS5)
│       └── yolo_strips.onnx          # Trained YOLO26s ONNX blister-strip detector weights
├── models/
│   ├── Modelfile                     # Ollama Modelfile for oushudhcheck-gemma3-1b-bn.Q4_K_M.gguf
│   └── system_prompt.txt             # Trained grounded Bangla clinical system prompt
├── ml_training/
│   └── notebooks/
│       ├── make_dataset.ipynb                    # Joins & cleans 6 raw Kaggle MedEx BD CSV tables (21,714 rows)
│       ├── train_yolo26s_blister_strips.ipynb    # Trains YOLO26s on public + 3,000 synthetic BD foil scenes -> ONNX
│       └── train-model-oshudcheck.ipynb          # Fine-tunes Gemma 3 1B on medicines.db -> Q4_K_M GGUF (768.7 MB)
├── scripts/
│   ├── ingest_drugs.py               # Builds deduplicated data/processed/medicines.db + FTS5 index
│   ├── download_model.sh             # Auto-downloads & extracts Gemma 3 1B GGUF from GitHub Releases
│   └── benchmark.py                  # Evaluates all real Bangladeshi strip photos in test/
├── test/
│   ├── newimg (1).jpg .. (3).jpg     # Real 3-strip, 4-strip & 5-strip Bangladeshi tabletop benchmark photos
│   ├── img1.jpg, img (1)..(5).jpeg   # Reflective foil, vertical 90°, & 4-ingredient combination strip photos
│   └── caregiver_voice_examples/     # 4 ready-to-test Bangla caregiver voice follow-up MP3 clips
├── tests/
│   └── test_caregiver_qa.py          # Automated verification suite for Challenge 1 & Challenge 2
├── docs/designs/
│   └── yolo26-hybrid-strip-detector.md # Architectural design record for 3-stage YOLO26s + OCR hybrid fusion
├── docker-compose.yml                # One-command 5-service full-stack deployment
├── Dockerfile                        # Python 3.12-slim container for FastAPI backend & Telegram bot
├── run_local.bat                     # One-click Windows local full-stack launcher
├── agent.md                          # Authoritative AI agent & developer engineering guide
└── project.pdf                       # Original competition concept proposal
```

---

## DualStack (`OushudhCheck`)

| Team Member | Role | Focus Areas | Links |
| :--- | :--- | :--- | :--- |
| **Jubayer Ahmed** | **Frontend & AI Product Engineer** | Next.js 16 App Router architecture, Multi-Strip Scanner Workspace (`/scan`), 6-Stage AI Transparency Inspector (`/pipeline`), 17,191 Medicine Directory (`/database`), and accessibility UX. | GitHub: [@Jubayerx17](https://github.com/Jubayerx17)<br>Email: `jubayer.ahm17@gmail.com` |
| **Ridoy Hossan Rony** | **Backend & ML Systems Engineer** | YOLO26s ONNX Hybrid Strip Detector, RapidOCR PP-OCRv6 + LAB-CLAHE pipeline, 17,191 BD Medicine Resolver, Fine-Tuned Gemma 3 1B Bangla Safety SLM, Caregiver Mode Q&A, and Telegram Bot (`@OushudhCheck_bot`). | GitHub: [@rhridoyhossan](https://github.com/rhridoyhossan)<br>Email: `rhridoyhossan@gmail.com` |

---

## Important Medical Disclaimer

OushudhCheck is an AI-assisted medication identification, duplicate-ingredient screening, and Bangla patient-education platform grounded in structured Bangladeshi pharmaceutical data (`medicines.db`). It is designed to assist patients, family caregivers, community health workers, and pharmacists, and **must not** be treated as a substitute for a licensed physician's prescription, diagnosis, or clinical judgment. Always consult a qualified healthcare professional before starting, stopping, or modifying any medication regimen. For 24/7 government medical consultation in Bangladesh, call **স্বাস্থ্য বাতায়ন (`16263`)** or **জাতীয় জরুরি সেবা (`999`)**.

---

<p align="center">
  <strong>DualStack: OushudhCheck: Know Your Medicines. Protect Your Health.</strong>
</p>
