"""
Automated tests for Caregiver Q&A follow-up answers (SLM prompt/validation + deterministic fallback)
and regression checks for initial medicine scan/lookup summaries.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from backend.app.models.schemas import MedicineInfo, SafetyAlert, StripScanResult
from backend.app.services import slm
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
        pregnancy_and_lactation_description="Avoid during pregnancy unless clearly needed.",
        precautions_description="May cause drowsiness; avoid driving or operating machinery.",
        storage_conditions_description="Store below 30 C in a cool dry place.",
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
        side_effects_description="Gastric irritation, nausea, heartburn, headache.",
        interaction_description="Avoid combining with other NSAIDs or anticoagulants.",
        contraindications_description="Active peptic ulcer, severe renal impairment.",
        pregnancy_and_lactation_description="Avoid in third trimester of pregnancy.",
        precautions_description="Take after food; use caution in ulcer, kidney disease, or asthma.",
        storage_conditions_description="Keep away from light and moisture below 30 C.",
    )
    return StripScanResult(
        strip_index=1,
        raw_ocr_text="Naprosyn 500 mg Naproxen Tablet",
        extracted_strength="500 mg",
        detected_dosage_form="Tablet",
        matched_medicine=med,
    )


def _make_seclo_strip() -> StripScanResult:
    med = MedicineInfo(
        brand_id=303,
        brand_name="Seclo",
        generic="Omeprazole",
        dosage_form="Capsule",
        manufacturer="Square Pharmaceuticals PLC",
        package_container="20 mg",
        drug_class="Proton Pump Inhibitor",
        indication="GERD, Peptic ulcer, Acidity",
        dosage_description="20 mg once daily 30 minutes before meal.",
    )
    return StripScanResult(
        strip_index=1,
        raw_ocr_text="Seclo 20 mg Omeprazole Capsule",
        extracted_strength="20 mg",
        detected_dosage_form="Capsule",
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
    def setUp(self) -> None:
        with slm._SLM_CACHE_LOCK:
            slm._SLM_QA_CACHE.clear()
            slm._SLM_CIRCUIT_OPEN_UNTIL = 0.0

    # --- 1. User's 3 exact examples (deterministic / offline SLM path) ---

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
        self.assertNotIn("সতর্কতা:", ans)
        self.assertNotIn("Clonazepam, ট্যাবলেট", ans)
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
        self.assertNotIn("সতর্কতা:", ans_purpose)
        self.assertNotIn("রাতে ঘুমানোর আগে", ans_purpose)
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
        self.assertNotIn("সতর্কতা:", ans)
        self.assertNotIn("Adults: 500 mg", ans)
        self.assertNotIn("Naproxen, ট্যাবলেট", ans)
        self.assertLess(len(ans), 180)

    # --- 2. Additional follow-up intents ---

    def test_other_common_intents_are_concise_and_targeted(self) -> None:
        epitra = _make_epitra_strip()
        naprosyn = _make_naprosyn_strip()
        with patch.object(slm, "SLM_API_URL", ""):
            ans_side = slm.answer_caregiver_followup("এর পার্শ্বপ্রতিক্রিয়া কী?", [epitra])
            self.assertIn("পার্শ্বপ্রতিক্রিয়া", ans_side)
            self.assertNotIn("হলো একটি", ans_side)
            self.assertNotIn("সতর্কতা:", ans_side)

            ans_warn = slm.answer_caregiver_followup("এই ওষুধের সতর্কতা কী?", [naprosyn])
            self.assertIn("সতর্কতা:", ans_warn)
            self.assertNotIn("হলো একটি", ans_warn)

            ans_preg = slm.answer_caregiver_followup("গর্ভাবস্থায় এটি কি খাওয়া যাবে?", [epitra])
            self.assertIn("গর্ভাবস্থা", ans_preg)
            self.assertNotIn("হলো একটি", ans_preg)

            ans_dose = slm.answer_caregiver_followup("দিনে কয়বার খেতে হবে?", [naprosyn])
            self.assertIn("মাত্রা", ans_dose)
            self.assertNotIn("হলো একটি", ans_dose)
            self.assertNotIn("সতর্কতা:", ans_dose)

            ans_ing = slm.answer_caregiver_followup("এর জেনেরিক উপাদান কী?", [epitra])
            self.assertIn("Clonazepam", ans_ing)
            self.assertNotIn("হলো একটি", ans_ing)
            self.assertNotIn("সতর্কতা:", ans_ing)

            ans_inter = slm.answer_caregiver_followup("অন্য ওষুধের সাথে একসাথে খাওয়া যাবে কি?", [naprosyn])
            self.assertIn("বিক্রিয়া", ans_inter)
            self.assertNotIn("হলো একটি", ans_inter)

            ans_unknown = slm.answer_caregiver_followup("আচ্ছা বলো তো?", [epitra])
            self.assertNotIn("হলো একটি", ans_unknown)
            self.assertNotIn("মূল সক্রিয় উপাদান", ans_unknown)
            self.assertNotIn("সতর্কতা:", ans_unknown)

    # --- 3. SLM summary-regurgitation rejection & valid SLM pass-through ---

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

    # --- 4. Non-oral route safety & multi-strip Q&A ---

    def test_non_oral_suppository_never_gets_oral_timing(self) -> None:
        supp = _make_suppository_strip()
        with patch.object(slm, "SLM_API_URL", ""):
            ans = slm.answer_caregiver_followup("এটি কি খাবারের আগে না পরে খাবে?", [supp])
        self.assertIn("পায়ুপথে", ans)
        self.assertIn("মুখে খাওয়া যাবে না", ans)
        self.assertNotIn("খাবারের পরে", ans)

    def test_multi_strip_interaction_alert_preserved(self) -> None:
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

    # --- 5. Initial scan/lookup full summary NOT regressed ---

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

    # --- 6. Telegram bot follow-up routing ---

    def test_telegram_followup_detector_matches_reported_questions(self) -> None:
        self.assertTrue(_looks_like_followup_question("এই মেডিসিন টি কি আমি সকালে খেতে পারবো"))
        self.assertTrue(_looks_like_followup_question("এই মেডিসিনের কাজ কি?"))
        self.assertTrue(_looks_like_followup_question("এই ওষুধটি কি খাবারের আগে না পরে খাবে"))
        self.assertTrue(_looks_like_followup_question("সকালে খেতে পারবো"))
        self.assertFalse(_looks_like_followup_question("Napa Extra"))


if __name__ == "__main__":
    unittest.main()
