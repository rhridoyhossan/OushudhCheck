"""
Automated tests for DualStack OushudhCheck Onsite Challenges:
1. Challenge 1: Caregiver Mode Bangla Voice/Text Follow-Up Q&A (`answer_caregiver_followup`):
   - 100% Bangla answers with zero English prose leaks (`_has_english_prose_leak == False`) across all 9 intents.
   - Question-specific answers without dumping raw English `medicines.db` paragraphs, unsolicited ingredient introductions,
     or unrelated warnings/meal timing on focused pregnancy & lactation questions.
   - Non-oral route safety guardrails (`_enforce_route_safety`) for Suppository, Cream, Drops, Injection, and Inhaler.
   - Multi-strip focus selection (`_select_focused_strips`) and clinical grounding validator (`_is_slm_qa_answer_clinically_valid`).
2. Challenge 2: Trustworthy Safety Alerts on Blurry/Noisy OCR, Unreadable Brands & Similar Generic Names:
   - Torn-foil substring ghost brand suppression (`SafetySLMEngine.filter_ghost_and_prefix_brands`: `'dipine'`, `'epam'`).
   - Multi-word brand upgrade (`'napa'` + `'extra'` -> `'napa extra'`).
   - Deterministic duplicate active ingredient & interaction detection (`evaluate_multi_strip_safety`: `Napa` + `Ace`, `Napa` + `Napa Extra`, safe distinct generics).
   - Active-ingredient strength refinement (`SafetySLMEngine.refine_strength_for_medicine`) and unreadable-brand generic/EDCL fallback.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from backend.app.models.schemas import MedicineInfo, SafetyAlert, StripScanResult
from backend.app.services import ocr, slm
from backend.app.services.ocr import (
    MedicineMatcher,
     evaluate_multi_strip_safety,
    extract_strengths,
)
from backend.app.services.slm import SafetySLMEngine
from bot.telegram_bot import _looks_like_followup_question


def _make_epitra_strip() -> StripScanResult:
    med = MedicineInfo(
        brand_id=101,
        brand_name="Epitra",
        generic="Clonazepam",
        dosage_form="Tablet",
        manufacturer="Incepta Pharmaceuticals Ltd.",
        package_container="1 mg",
        drug_class="Benzodiazepine",
        indication="Panic disorder, Epilepsy, Anxiety",
        indication_description="Indicated for panic disorder, seizures, and anxiety disorders.",
        dosage_description="Adults: 0.5 mg to 2 mg daily in divided doses or at bedtime.",
        side_effects_description="Drowsiness, dizziness, fatigue, coordination impairment.",
        interaction_description="CNS depressants and alcohol increase sedation.",
        contraindications_description="Severe hepatic impairment, acute narrow-angle glaucoma.",
        pregnancy_and_lactation_description="Avoid during pregnancy unless clearly needed. Excreted in breast milk; avoid during lactation.",
        precautions_description="May cause drowsiness; avoid driving or operating machinery.",
        storage_conditions_description="Store below 30 C in a cool dry place away from light and moisture.",
    )
    return StripScanResult(
        strip_index=1,
        raw_ocr_text="Epitra 1 mg Clonazepam Tablet",
        extracted_strength="1 mg",
        detected_dosage_form="Tablet",
        matched_medicine=med,
    )


def _make_naprosyn_strip() -> StripScanResult:
    med = MedicineInfo(
        brand_id=202,
        brand_name="Naprosyn",
        generic="Naproxen",
        dosage_form="Tablet",
        manufacturer="Radiant Pharmaceuticals Ltd.",
        package_container="500 mg",
        drug_class="NSAID",
        indication="Pain, inflammation, migraine, arthritis",
        indication_description="Used to relieve pain, tenderness, swelling, and stiffness.",
        dosage_description="Adults: 500 mg twice daily with food or after meal.",
        side_effects_description="Gastric irritation, nausea, heartburn, headache, dizziness.",
        interaction_description="Avoid combining with other NSAIDs, aspirin, or anticoagulants such as warfarin.",
        contraindications_description="Active peptic ulcer, severe renal impairment.",
        pregnancy_and_lactation_description="Avoid in third trimester of pregnancy. Avoid during breast-feeding.",
        precautions_description="Take after food; use caution in peptic ulcer, kidney disease, or asthma.",
        storage_conditions_description="Keep away from light and moisture below 30 C.",
    )
    return StripScanResult(
        strip_index=1,
        raw_ocr_text="Naprosyn 500 mg Naproxen Tablet",
        extracted_strength="500 mg",
        detected_dosage_form="Tablet",
        matched_medicine=med,
    )


def _make_seclo_strip(strip_index: int = 1) -> StripScanResult:
    med = MedicineInfo(
        brand_id=303,
        brand_name="Seclo",
        generic="Omeprazole",
        dosage_form="Capsule",
        manufacturer="Square Pharmaceuticals PLC",
        package_container="20 mg",
        drug_class="Proton Pump Inhibitor",
        indication="GERD, Peptic ulcer, Acidity",
        indication_description="Treatment of gastroesophageal reflux disease and peptic ulcer.",
        dosage_description="20 mg once or twice daily 30 minutes before meal.",
        side_effects_description="Headache, diarrhea, abdominal pain, nausea, flatulence.",
        interaction_description="May interact with clopidogrel, diazepam, phenytoin, and warfarin.",
        pregnancy_and_lactation_description="Category C; use only if clearly needed during pregnancy.",
        precautions_description="Exclude gastric malignancy before long-term hepatic use.",
        storage_conditions_description="Store below 25 C in a dry place protected from light.",
    )
    return StripScanResult(
        strip_index=strip_index,
        raw_ocr_text="Seclo 20 mg Omeprazole Capsule",
        extracted_strength="20 mg",
        detected_dosage_form="Capsule",
        matched_medicine=med,
    )


def _make_napa_strip(strip_index: int = 1) -> StripScanResult:
    med = MedicineInfo(
        brand_id=401,
        brand_name="Napa",
        generic="Paracetamol",
        dosage_form="Tablet",
        manufacturer="Beximco Pharmaceuticals Ltd.",
        package_container="500 mg",
        drug_class="Analgesic & Antipyretic",
        indication="Fever, headache, toothache, body pain",
        dosage_description="Adults: 500 mg to 1000 mg every 4 to 6 hours after food.",
        side_effects_description="Skin rash, allergic reactions, rare hepatotoxicity in overdose.",
        interaction_description="Alcohol and anticonvulsants may increase liver toxicity risk.",
        pregnancy_and_lactation_description="Category B; generally safe in pregnancy at recommended doses.",
        precautions_description="Use caution in severe hepatic or renal impairment.",
        storage_conditions_description="Store below 30 C in a cool and dry place.",
    )
    return StripScanResult(
        strip_index=strip_index,
        raw_ocr_text="Napa 500 mg Paracetamol Tablet",
        extracted_strength="500 mg",
        detected_dosage_form="Tablet",
        matched_medicine=med,
    )


def _make_napa_extra_strip(strip_index: int = 2) -> StripScanResult:
    med = MedicineInfo(
        brand_id=402,
        brand_name="Napa Extra",
        generic="Paracetamol + Caffeine",
        dosage_form="Tablet",
        manufacturer="Beximco Pharmaceuticals Ltd.",
        package_container="500 mg + 65 mg",
        drug_class="Analgesic & Antipyretic",
        indication="Fever, migraine, severe headache, dental pain",
        dosage_description="Adults: 1-2 tablets three times daily after meal.",
        side_effects_description="Insomnia, palpitations, nausea, dizziness.",
        interaction_description="Avoid taking with other paracetamol-containing products.",
        pregnancy_and_lactation_description="Avoid excessive caffeine during pregnancy and lactation.",
        precautions_description="Caution in hepatic impairment or severe hypertension.",
        storage_conditions_description="Store below 30 C away from light and moisture.",
    )
    return StripScanResult(
        strip_index=strip_index,
        raw_ocr_text="Napa Extra 500 mg + 65 mg Tablet",
        extracted_strength="500 mg + 65 mg",
        detected_dosage_form="Tablet",
        matched_medicine=med,
    )


def _make_ace_strip(strip_index: int = 2) -> StripScanResult:
    med = MedicineInfo(
        brand_id=403,
        brand_name="Ace",
        generic="Paracetamol",
        dosage_form="Tablet",
        manufacturer="Square Pharmaceuticals PLC",
        package_container="500 mg",
        drug_class="Analgesic & Antipyretic",
        indication="Fever and mild to moderate pain",
        dosage_description="1 tablet 3 to 4 times daily after meal.",
    )
    return StripScanResult(
        strip_index=strip_index,
        raw_ocr_text="Ace 500 mg Paracetamol Tablet",
        extracted_strength="500 mg",
        detected_dosage_form="Tablet",
        matched_medicine=med,
    )


def _make_suppository_strip() -> StripScanResult:
    med = MedicineInfo(
        brand_id=404,
        brand_name="Napa",
        generic="Paracetamol",
        dosage_form="Suppository",
        manufacturer="Beximco Pharmaceuticals Ltd.",
        package_container="125 mg",
        drug_class="Analgesic & Antipyretic",
        indication="Fever and pain",
        dosage_description="Tablet: 500 mg orally every 6 hours. Suppository: insert rectally every 6 hours.",
    )
    return StripScanResult(
        strip_index=1,
        raw_ocr_text="Napa 125 mg Suppository",
        extracted_strength="125 mg",
        detected_dosage_form="Suppository",
        matched_medicine=med,
    )


def _make_non_oral_strip(dosage_form: str, brand_name: str = "Dermacare", generic: str = "Clotrimazole") -> StripScanResult:
    med = MedicineInfo(
        brand_id=505,
        brand_name=brand_name,
        generic=generic,
        dosage_form=dosage_form,
        manufacturer="Square Pharmaceuticals PLC",
        package_container="10 gm",
        drug_class="Antifungal",
        indication="Fungal infection",
        dosage_description="Apply or administer as directed by physician.",
    )
    return StripScanResult(
        strip_index=1,
        raw_ocr_text=f"{brand_name} {dosage_form}",
        extracted_strength="10 gm",
        detected_dosage_form=dosage_form,
        matched_medicine=med,
    )


class _FakeHTTPResponse:
    def __init__(self, content: str) -> None:
        self._body = json.dumps(
            {"choices": [{"message": {"content": content}}]},
            ensure_ascii=False,
        ).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *args) -> None:
        return None


class CaregiverQATests(unittest.TestCase):
    """Challenge 1: Caregiver Mode Bangla Follow-Up Q&A Verification Suite."""

    def setUp(self) -> None:
        with slm._SLM_CACHE_LOCK:
            slm._SLM_QA_CACHE.clear()
            slm._SLM_CIRCUIT_OPEN_UNTIL = 0.0

    # --- 1. User's exact reported questions & 100% pure Bangla enforcement ---

    def test_epitra_morning_question_concise_and_specific(self) -> None:
        strip = _make_epitra_strip()
        with patch.object(slm, "SLM_API_URL", ""):
            ans = slm.answer_caregiver_followup(
                "এই মেডিসিন টি কি আমি সকালে খেতে পারবো",
                [strip],
            )
        self.assertIn("রাতে", ans)
        self.assertIn("সকালে", ans)
        self.assertNotIn("হলো একটি", ans)
        self.assertNotIn("মূল সক্রিয় উপাদান", ans)
        self.assertNotIn("মূল জেনেরিক উপাদান", ans)
        self.assertNotIn("সতর্কতা:", ans)
        self.assertFalse(slm._has_english_prose_leak(ans, [strip]))
        self.assertLess(len(ans), 240)

    def test_epitra_purpose_question_concise_and_distinct(self) -> None:
        strip = _make_epitra_strip()
        with patch.object(slm, "SLM_API_URL", ""):
            ans_morning = slm.answer_caregiver_followup(
                "এই মেডিসিন টি কি আমি সকালে খেতে পারবো",
                [strip],
            )
            ans_purpose = slm.answer_caregiver_followup(
                "এই মেডিসিনের কাজ কি?",
                [strip],
            )
        self.assertNotEqual(ans_morning, ans_purpose)
        self.assertIn("ব্যবহৃত হয়", ans_purpose)
        self.assertTrue(any(w in ans_purpose for w in ("উদ্বেগ", "অনিদ্রা", "খিঁচুনি", "Anxiety")))
        self.assertNotIn("হলো একটি", ans_purpose)
        self.assertNotIn("মূল সক্রিয় উপাদান", ans_purpose)
        self.assertNotIn("মূল জেনেরিক উপাদান", ans_purpose)
        self.assertNotIn("সতর্কতা:", ans_purpose)
        self.assertNotIn("রাতে ঘুমানোর আগে", ans_purpose)
        self.assertFalse(slm._has_english_prose_leak(ans_purpose, [strip]))
        self.assertLess(len(ans_purpose), 220)

    def test_naprosyn_before_or_after_meal_concise_and_specific(self) -> None:
        strip = _make_naprosyn_strip()
        with patch.object(slm, "SLM_API_URL", ""):
            ans = slm.answer_caregiver_followup(
                "এই ওষুধটি কি খাবারের আগে না পরে খাবে",
                [strip],
            )
        self.assertIn("খাবারের পরে", ans)
        self.assertNotIn("হলো একটি", ans)
        self.assertNotIn("মূল সক্রিয় উপাদান", ans)
        self.assertNotIn("মূল জেনেরিক উপাদান", ans)
        self.assertNotIn("সতর্কতা:", ans)
        self.assertNotIn("Adults: 500 mg", ans)
        self.assertFalse(slm._has_english_prose_leak(ans, [strip]))
        self.assertLess(len(ans), 180)

    def test_dosage_how_to_take_bangla_only_without_ingredient_dump(self) -> None:
        """
        Verifies that 'ওষুধ নিয়ে আমি কিভাবে খাব' across Naprosyn 500 mg, Seclo 20 mg,
        and Napa Extra returns 100% Bangla dosing guidance without leaking English prose
        or dumping 'এর মূল জেনেরিক উপাদান হলো ...'.
        """
        strips_and_expected = [
            (_make_naprosyn_strip(), "খাবারের পরে"),
            (_make_seclo_strip(), "খাবারের ৩০ মিনিট আগে"),
            (_make_napa_extra_strip(strip_index=1), "খাবারের পরে"),
        ]
        with patch.object(slm, "SLM_API_URL", ""):
            for strip, expected_phrase in strips_and_expected:
                ans = slm.answer_caregiver_followup("ওষুধ নিয়ে আমি কিভাবে খাব", [strip])
                self.assertIn(expected_phrase, ans)
                self.assertFalse(
                    slm._has_english_prose_leak(ans, [strip]),
                    f"English leak detected for {strip.matched_medicine.brand_name}: {ans}",
                )
                self.assertNotIn("মূল জেনেরিক উপাদান", ans)
                self.assertNotIn("মূল সক্রিয় উপাদান", ans)
                self.assertNotIn("হলো একটি", ans)
                self.assertNotIn("Adults:", ans)
                self.assertNotIn("twice daily", ans.lower())

    # --- 2. All 9 intents: 100% Bangla & zero English prose leaks ---

    def test_all_nine_intents_return_pure_bangla_without_english_leaks(self) -> None:
        epitra = _make_epitra_strip()
        naprosyn = _make_naprosyn_strip()
        seclo = _make_seclo_strip()

        intent_cases = [
            ("dosage", "দিনে কয়বার এবং কিভাবে খেতে হবে?", naprosyn, "দিনে ২ বার"),
            ("meal_timing", "খাবারের আগে না পরে কখন খাব?", seclo, "খাবারের ৩০ মিনিট আগে"),
            ("side_effects", "এর পার্শ্বপ্রতিক্রিয়া কী কী?", naprosyn, "পার্শ্বপ্রতিক্রিয়া"),
            ("pregnancy_lactation", "গর্ভাবস্থায় নিরাপদ কি?", naprosyn, "গর্ভাবস্থায়"),
            ("interactions_duplicates", "অন্য ওষুধের সাথে একসাথে খাওয়া যাবে কি?", naprosyn, "বিক্রিয়া"),
            ("storage", "এই ওষুধ কোথায় এবং কীভাবে সংরক্ষণ করব?", seclo, "সংরক্ষণ"),
            ("warnings_precautions", "এই ওষুধের সতর্কতা ও নিষেধাজ্ঞা কী?", naprosyn, "সতর্কতা:"),
            ("purpose", "এই মেডিসিনের কাজ কি?", seclo, "গ্যাস্ট্রিক"),
            ("ingredients", "এর জেনেরিক উপাদান কী?", epitra, "Clonazepam"),
        ]

        with patch.object(slm, "SLM_API_URL", ""):
            for intent_name, question, strip, expected_sub in intent_cases:
                detected = slm._detect_caregiver_intents(question)
                self.assertIn(intent_name, detected, f"Expected intent {intent_name} for {question!r}")
                ans = slm.answer_caregiver_followup(question, [strip])
                self.assertIn(expected_sub, ans, f"Missing {expected_sub!r} in {intent_name} answer: {ans}")
                self.assertFalse(
                    slm._has_english_prose_leak(ans, [strip]),
                    f"English prose leak on intent={intent_name}: {ans}",
                )
                self.assertNotIn("হলো একটি", ans)
                if intent_name != "ingredients":
                    self.assertNotIn("মূল জেনেরিক উপাদান", ans)
                    self.assertNotIn("মূল সক্রিয় উপাদান", ans)

    def test_focused_pregnancy_and_lactation_questions_do_not_append_unrelated_intents(self) -> None:
        """
        Ensures 'গর্ভাবস্থায় নিরাপদ কি?' does not trigger general warnings ('সতর্কতা:')
        and 'বুকের দুধ খাওয়ানোর সময় খাওয়া যাবে?' does not trigger meal timing ('খাবারের পরে').
        """
        naprosyn = _make_naprosyn_strip()
        with patch.object(slm, "SLM_API_URL", ""):
            preg_q = "গর্ভাবস্থায় নিরাপদ কি?"
            self.assertEqual(slm._detect_caregiver_intents(preg_q), ["pregnancy_lactation"])
            ans_preg = slm.answer_caregiver_followup(preg_q, [naprosyn])
            self.assertIn("গর্ভাবস্থায়", ans_preg)
            self.assertNotIn("সতর্কতা:", ans_preg)
            self.assertNotIn("খাবারের পরে", ans_preg)
            self.assertFalse(slm._has_english_prose_leak(ans_preg, [naprosyn]))

            lact_q = "বুকের দুধ খাওয়ানোর সময় খাওয়া যাবে?"
            self.assertEqual(slm._detect_caregiver_intents(lact_q), ["pregnancy_lactation"])
            ans_lact = slm.answer_caregiver_followup(lact_q, [naprosyn])
            self.assertIn("স্তন্যদানকালে", ans_lact)
            self.assertNotIn("খাবারের পরে", ans_lact)
            self.assertNotIn("সতর্কতা:", ans_lact)
            self.assertFalse(slm._has_english_prose_leak(ans_lact, [naprosyn]))

    # --- 3. SLM validation, English-leak rejection & summary-regurgitation rejection ---

    def test_slm_regurgitated_summary_is_rejected_and_falls_back(self) -> None:
        epitra = _make_epitra_strip()
        regurgitated = (
            "Epitra (1 mg) হলো একটি ট্যাবলেট, যার মূল সক্রিয় উপাদান Clonazepam। "
            "এটি মানসিক উদ্বেগ ও খিঁচুনি নিয়ন্ত্রণে ব্যবহৃত হয়। এটি খাবারের পরে খেতে হয়।\n"
            "সতর্কতা: ঘুম ঘুম ভাব হতে পারে।"
        )
        with patch.object(slm, "SLM_API_URL", "http://127.0.0.1:8081/v1/chat/completions"), patch(
            "urllib.request.urlopen", return_value=_FakeHTTPResponse(regurgitated)
        ):
            ans = slm.answer_caregiver_followup("এই মেডিসিন টি কি আমি সকালে খেতে পারবো", [epitra])

        self.assertNotIn("হলো একটি", ans)
        self.assertNotIn("সতর্কতা:", ans)
        self.assertIn("রাতে", ans)
        self.assertIn("সকালে", ans)

    def test_slm_english_prose_leak_and_unsolicited_ingredient_dump_rejected(self) -> None:
        naprosyn = _make_naprosyn_strip()

        leaked_english = "এই ওষুধটি Adults: 500 mg twice daily with food or after meal হিসেবে খেতে হবে।"
        self.assertTrue(slm._has_english_prose_leak(leaked_english, [naprosyn]))
        self.assertFalse(
            slm._is_slm_qa_answer_clinically_valid(
                leaked_english, [naprosyn], [], ["dosage"], question="ওষুধ নিয়ে আমি কিভাবে খাব"
            )
        )

        unsolicited_ingredient = "এর মূল জেনেরিক উপাদান হলো Naproxen। এটি দিনে ২ বার খাবারের পরে খেতে হয়।"
        self.assertFalse(
            slm._is_slm_qa_answer_clinically_valid(
                unsolicited_ingredient, [naprosyn], [], ["dosage"], question="ওষুধ নিয়ে আমি কিভাবে খাব"
            )
        )

        with patch.object(slm, "SLM_API_URL", "http://127.0.0.1:8081/v1/chat/completions"), patch(
            "urllib.request.urlopen", return_value=_FakeHTTPResponse(leaked_english)
        ):
            ans = slm.answer_caregiver_followup("ওষুধ নিয়ে আমি কিভাবে খাব", [naprosyn])
        self.assertFalse(slm._has_english_prose_leak(ans, [naprosyn]))
        self.assertIn("দিনে ২ বার", ans)

    def test_slm_unsolicited_warning_rejected_for_meal_timing(self) -> None:
        naprosyn = _make_naprosyn_strip()
        bad_with_warning = "এটি খাবারের পরে খেতে হবে।\nসতর্কতা: কিডনি রোগ থাকলে খাবেন না।"
        with patch.object(slm, "SLM_API_URL", "http://127.0.0.1:8081/v1/chat/completions"), patch(
            "urllib.request.urlopen", return_value=_FakeHTTPResponse(bad_with_warning)
        ):
            ans = slm.answer_caregiver_followup("এই ওষুধটি কি খাবারের আগে না পরে খাবে", [naprosyn])
        self.assertNotIn("সতর্কতা:", ans)
        self.assertIn("খাবারের পরে", ans)

    def test_slm_concise_valid_answer_is_accepted_and_scoped_prompt_used(self) -> None:
        naprosyn = _make_naprosyn_strip()
        good_slm = "Naprosyn অবশ্যই খাবারের পরে (ভরা পেটে) খেতে হবে যাতে পেটে জ্বালাপোড়া না হয়।"
        captured_payloads = []

        def _fake_urlopen(req, timeout=30.0):
            captured_payloads.append(json.loads(req.data.decode("utf-8")))
            return _FakeHTTPResponse(good_slm)

        with patch.object(slm, "SLM_API_URL", "http://127.0.0.1:8081/v1/chat/completions"), patch(
            "urllib.request.urlopen", side_effect=_fake_urlopen
        ):
            ans = slm.answer_caregiver_followup("এই ওষুধটি কি খাবারের আগে না পরে খাবে", [naprosyn])

        self.assertEqual(ans, good_slm)
        self.assertEqual(len(captured_payloads), 1)
        prompt_text = captured_payloads[0]["messages"][0]["content"]
        self.assertIn("[ওষুধের প্রাসঙ্গিক তথ্য]", prompt_text)
        self.assertNotIn("পার্শ্বপ্রতিক্রিয়া:", prompt_text)
        self.assertNotIn("সংরক্ষণ:", prompt_text)
        self.assertNotIn("জেনেরিক উপাদান:", prompt_text)
        self.assertLessEqual(captured_payloads[0]["max_tokens"], 180)

    def test_slm_bedtime_sedative_morning_hallucination_rejected(self) -> None:
        epitra = _make_epitra_strip()
        unsafe_slm = "হ্যাঁ, আপনি সকালে খেতে পারবেন।"
        with patch.object(slm, "SLM_API_URL", "http://127.0.0.1:8081/v1/chat/completions"), patch(
            "urllib.request.urlopen", return_value=_FakeHTTPResponse(unsafe_slm)
        ):
            ans = slm.answer_caregiver_followup("এই মেডিসিন টি কি আমি সকালে খেতে পারবো", [epitra])
        self.assertNotEqual(ans, unsafe_slm)
        self.assertIn("রাতে", ans)
        self.assertTrue("ঘুম" in ans or "তন্দ্রা" in ans)

    def test_slm_short_spaced_bangla_reply_accepted(self) -> None:
        epitra = _make_epitra_strip()
        short_slm = "হ্যাঁ, এটি রাতে ঘুমানোর আগে খেতে হয়।"
        with patch.object(slm, "SLM_API_URL", "http://127.0.0.1:8081/v1/chat/completions"), patch(
            "urllib.request.urlopen", return_value=_FakeHTTPResponse(short_slm)
        ):
            ans = slm.answer_caregiver_followup("এটি কি রাতে খাওয়া যাবে?", [epitra])
        self.assertEqual(ans, short_slm)

    # --- 4. Non-oral route safety across all 5 non-oral dosage forms ---

    def test_non_oral_routes_never_get_oral_meal_timing(self) -> None:
        non_oral_cases = [
            (_make_suppository_strip(), "পায়ুপথে", "মুখে খাওয়া যাবে না"),
            (_make_non_oral_strip("Cream", "Pevisone", "Econazole"), "বাহ্যিক ব্যবহারের", "মুখে খাওয়া যাবে না"),
            (_make_non_oral_strip("Ophthalmic Drop", "Norsol", "Sodium Chloride"), "ড্রপ হিসেবে", "মুখে খাওয়া যাবে না"),
            (_make_non_oral_strip("IM/IV Injection", "Ceftriaxone", "Ceftriaxone"), "ইনজেকশন", "তত্ত্বাবধানে"),
            (_make_non_oral_strip("Inhaler", "Brodil", "Salbutamol"), "শ্বাসের মাধ্যমে", "ইনহেলার"),
        ]
        with patch.object(slm, "SLM_API_URL", ""):
            for strip, expected_kw1, expected_kw2 in non_oral_cases:
                ans = slm.answer_caregiver_followup("এটি কি খাবারের আগে না পরে কীভাবে খাব?", [strip])
                self.assertIn(expected_kw1, ans)
                self.assertIn(expected_kw2, ans)
                self.assertNotIn("খাবারের পরে", ans)
                self.assertNotIn("ভরা পেটে", ans)
                self.assertNotIn("খালি পেটে", ans)

                # Also verify _enforce_route_safety strips hallucinated oral instructions directly
                hallucinated = "এটি প্রতিদিন খাবারের পরে ভরা পেটে পানি দিয়ে খেতে হয়।\nসতর্কতা: শিশুদের নাগালের বাইরে রাখুন।"
                sanitized = slm._enforce_route_safety(hallucinated, strip.detected_dosage_form or "")
                self.assertIn(expected_kw1, sanitized)
                self.assertNotIn("ভরা পেটে", sanitized)
                self.assertIn("সতর্কতা:", sanitized)

    # --- 5. Multi-strip focus selection & cross-strip interaction Q&A ---

    def test_multi_strip_focus_selection_and_combined_alerts(self) -> None:
        naprosyn = _make_naprosyn_strip()
        seclo = _make_seclo_strip(strip_index=2)
        strips = [naprosyn, seclo]

        # Naming one strip focuses only on that strip for meal_timing / dosage
        focused_seclo, all_valid = slm._select_focused_strips(
            "Seclo কি খাবারের আগে খাব?", strips, ["meal_timing"]
        )
        self.assertEqual(len(all_valid), 2)
        self.assertEqual(len(focused_seclo), 1)
        self.assertEqual(focused_seclo[0].matched_medicine.brand_name, "Seclo")

        # Fuzzy voice-ASR spelling ('naprosin') also focuses on Naprosyn
        focused_fuzzy, _ = slm._select_focused_strips(
            "naprosin কখন খাব?", strips, ["meal_timing"]
        )
        self.assertEqual(len(focused_fuzzy), 1)
        self.assertEqual(focused_fuzzy[0].matched_medicine.brand_name, "Naprosyn")

        # Asking about taking them together covers all strips
        focused_all, _ = slm._select_focused_strips(
            "দুইটা কি একসাথে খাওয়া যাবে?", strips, ["interactions_duplicates"]
        )
        self.assertEqual(len(focused_all), 2)

        # Duplicate alert preserved in caregiver follow-up answer
        n1 = _make_naprosyn_strip()
        n2 = _make_naprosyn_strip()
        n2.strip_index = 2
        alert = SafetyAlert(
            alert_type="duplicate_generic",
            severity="high",
            involved_strips=[1, 2],
            involved_brands=["Naprosyn", "Naprosyn"],
            generic_or_class="Naproxen",
            message_en="Duplicate Naproxen detected.",
            message_bn="উভয় পাতাতেই একই জেনেরিক Naproxen রয়েছে; একসাথে খাবেন না।",
        )
        with patch.object(slm, "SLM_API_URL", ""):
            ans = slm.answer_caregiver_followup("দুইটা কি একসাথে খাওয়া যাবে?", [n1, n2], [alert])
        self.assertIn("⚠️ নিরাপত্তা সতর্কতা:", ans)
        self.assertIn("একসাথে খাবেন না", ans)

    # --- 6. Initial scan/lookup full summary NOT regressed & Telegram routing ---

    def test_initial_beginner_summary_still_includes_full_structure(self) -> None:
        summary = slm.generate_bangla_beginner_summary(
            brand_name="Epitra",
            generic="Clonazepam",
            dosage_form="Tablet",
            manufacturer="Incepta Pharmaceuticals Ltd.",
            drug_class="Benzodiazepine",
            indication="Panic disorder, Epilepsy",
            strength="1 mg",
            use_llm=False,
        )
        self.assertIn("Epitra 1 mg হলো একটি ট্যাবলেট", summary)
        self.assertIn("মূল সক্রিয় উপাদান Clonazepam", summary)
        self.assertIn("সতর্কতা:", summary)

    def test_telegram_followup_detector_matches_reported_questions(self) -> None:
        self.assertTrue(_looks_like_followup_question("এই মেডিসিন টি কি আমি সকালে খেতে পারবো"))
        self.assertTrue(_looks_like_followup_question("এই মেডিসিনের কাজ কি?"))
        self.assertTrue(_looks_like_followup_question("এই ওষুধটি কি খাবারের আগে না পরে খাবে"))
        self.assertTrue(_looks_like_followup_question("ওষুধ নিয়ে আমি কিভাবে খাব"))
        self.assertTrue(_looks_like_followup_question("সকালে খেতে পারবো"))
        self.assertFalse(_looks_like_followup_question("Napa Extra"))


class BlurryNoisyOCRSafetyTests(unittest.TestCase):
    """Challenge 2: Trustworthy Safety Alerts on Blurry/Noisy OCR, Unreadable Brands & Similar Generics."""

    def test_torn_foil_substring_ghost_brand_suppression(self) -> None:
        """
        Ensures torn-foil substring ghosts like 'dipine' (from 'Amlodipine') or 'epam'
        (from 'Diazepam') are suppressed when the longer drug word is present in the image.
        """
        candidates = {"amdocal", "dipine", "epam"}
        ocr_words = {"amdocal", "amlodipine", "diazepam", "tablet", "5mg"}
        known_db = {"amdocal", "amlodipine", "diazepam", "dipine", "epam"}

        refined = SafetySLMEngine.filter_ghost_and_prefix_brands(
            brand_candidates=candidates,
            all_ocr_words_lower=ocr_words,
            full_ocr_text_lower="amdocal 5 amlodipine 5mg diazepam tablet",
            known_db_words=known_db,
        )
        self.assertIn("amdocal", refined)
        self.assertNotIn("dipine", refined)
        self.assertNotIn("epam", refined)

    def test_multi_word_brand_upgrade_napa_extra(self) -> None:
        """
        Ensures split foil boxes ('Napa' + 'Extra') upgrade cleanly to 'napa extra'
        and discard standalone 'napa' so caffeine is not missed.
        """
        candidates = {"napa", "seclo"}
        ocr_words = {"napa", "extra", "paracetamol", "caffeine", "seclo", "omeprazole"}
        known_db = {"napa", "napa extra", "seclo"}

        refined = SafetySLMEngine.filter_ghost_and_prefix_brands(
            brand_candidates=candidates,
            all_ocr_words_lower=ocr_words,
            full_ocr_text_lower="napa extra paracetamol 500 mg caffeine 65 mg seclo 20 mg",
            known_db_words=known_db,
        )
        self.assertIn("napa extra", refined)
        self.assertNotIn("napa", refined)
        self.assertIn("seclo", refined)

    def test_deterministic_duplicate_active_generic_detection(self) -> None:
        """
        Verifies 100% deterministic duplicate generic detection across different brands
        sharing Paracetamol (Napa + Ace, and Napa + Napa Extra multi-ingredient decomposition),
        while keeping distinct generics safe.
        """
        napa = _make_napa_strip(strip_index=1)
        ace = _make_ace_strip(strip_index=2)
        napa_extra = _make_napa_extra_strip(strip_index=2)
        seclo = _make_seclo_strip(strip_index=2)

        # Case 1: Napa (Paracetamol) + Ace (Paracetamol) -> High-severity duplicate_generic alert
        alerts_dup1, _, summary_bn1 = evaluate_multi_strip_safety([napa, ace])
        self.assertTrue(any(a.alert_type == "duplicate_generic" for a in alerts_dup1))
        dup_alert1 = next(a for a in alerts_dup1 if a.alert_type == "duplicate_generic")
        self.assertEqual(dup_alert1.severity, "high")
        self.assertEqual(dup_alert1.generic_or_class, "Paracetamol")
        self.assertEqual(dup_alert1.involved_strips, [1, 2])
        self.assertIn("Paracetamol", summary_bn1)

        # Case 2: Napa (Paracetamol) + Napa Extra (Paracetamol + Caffeine) -> High-severity duplicate Paracetamol alert
        alerts_dup2, _, _ = evaluate_multi_strip_safety([napa, napa_extra])
        self.assertTrue(
            any(a.alert_type == "duplicate_generic" and a.generic_or_class == "Paracetamol" for a in alerts_dup2)
        )

        # Case 3: Napa (Paracetamol) + Seclo (Omeprazole) -> Distinct generics, 0 alerts (all_safe)
        alerts_safe, _, summary_bn_safe = evaluate_multi_strip_safety([napa, seclo])
        self.assertEqual(alerts_safe, [])
        self.assertIn("কোনো ডুপ্লিকেট বা ঝুঁকিপূর্ণ আন্তঃক্রিয়া পাওয়া যায়নি", summary_bn_safe)

    def test_strength_refinement_for_single_and_combo_generics(self) -> None:
        """
        Verifies `SafetySLMEngine.refine_strength_for_medicine` aligns single-ingredient
        and multi-ingredient ('Paracetamol + Caffeine' -> '500 mg + 65 mg') strengths
        even when noisy OCR numbers appear in the cluster.
        """
        napa_extra_med = _make_napa_extra_strip().matched_medicine
        raw_combo_ocr = "Napa Extra\nParacetamol BP 500 mg &\nCaffeine USP 65 mg\nMfg Lic No 123"
        _, raw_detected = extract_strengths(raw_combo_ocr, brand_name="Napa Extra")
        primary, ordered = SafetySLMEngine.refine_strength_for_medicine(
            raw_cluster_text=raw_combo_ocr,
            all_detected_strengths=raw_detected,
            medicine=napa_extra_med,
            extract_strengths_fn=extract_strengths,
        )
        self.assertEqual(primary, "500 mg + 65 mg")
        self.assertEqual(ordered, ["500 mg", "65 mg"])

        seclo_med = _make_seclo_strip().matched_medicine
        raw_single_ocr = "Square Pharma\n30 Capsules\nSeclo 20 mg\nOmeprazole BP 20 mg"
        _, single_detected = extract_strengths(raw_single_ocr, brand_name="Seclo")
        primary_single, ordered_single = SafetySLMEngine.refine_strength_for_medicine(
            raw_cluster_text=raw_single_ocr,
            all_detected_strengths=single_detected,
            medicine=seclo_med,
            extract_strengths_fn=extract_strengths,
        )
        self.assertEqual(primary_single, "20 mg")
        self.assertEqual(ordered_single, ["20 mg"])

    def test_unreadable_brand_generic_and_edcl_fallback(self) -> None:
        """
        Verifies that when a strip's brand logo is torn or unreadable in a blurry photo,
        MedicineMatcher rescues the medicine via readable generic text ('Omeprazole USP 20 mg Capsule')
        or EDCL government generic strip registration ('DIAZEPAM TABLETS BP 5 mg').
        """
        matcher = MedicineMatcher()
        slm_engine = SafetySLMEngine()
        if not matcher._unique_brand_lowers:
            self.skipTest("medicines.db not present in test environment")

        # 1. Torn/unreadable brand rescued via generic_fallback
        res_generic = matcher.match_strip_text(
            raw_text="Each capsule contains Omeprazole USP 20 mg",
            strip_index=1,
            slm_engine=slm_engine,
        )
        self.assertIsNotNone(res_generic.matched_medicine)
        self.assertIn("omeprazole", res_generic.matched_medicine.generic.lower())
        self.assertEqual(res_generic.extracted_strength, "20 mg")

        # 2. EDCL government generic strip resolution
        res_edcl = matcher.match_strip_text(
            raw_text="Essential Drugs Co. Ltd.\nDIAZEPAM TABLETS BP 5 mg",
            strip_index=2,
            forced_brand_lower="diazepam",
            slm_engine=slm_engine,
        )
        self.assertIsNotNone(res_edcl.matched_medicine)
        self.assertEqual(res_edcl.matched_medicine.brand_name, "Diazepam")
        self.assertEqual(res_edcl.extracted_strength, "5 mg")

        # 3. Short/noisy garbage token rejected by fuzzy length-ratio & MIN_FUZZY_SCORE guard
        self.assertIsNone(matcher.resolve_brand_from_query("xy"))


if __name__ == "__main__":
    unittest.main()
