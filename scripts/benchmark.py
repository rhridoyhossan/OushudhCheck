r"""
Benchmark & Test-Folder Verification Runner for OushudhCheck.
Evaluates all real medicine strip photos in `test/` using RapidOCR (PP-OCRv6 ONNX) + RapidFuzz + medicines.db.
Run with:
    .\.venv\Scripts\python.exe scripts\benchmark.py
"""

from pathlib import Path
import sys
import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.ocr import MedicineStripScanner


def run_benchmark() -> None:
    scanner = MedicineStripScanner()
    test_dir = ROOT_DIR / "test"
    if not test_dir.exists():
        print(f"Test directory not found at {test_dir}; add strip photos there to run benchmark.")
        return

    print("=" * 95)
    print("OushudhCheck Real Test-Folder Image Benchmark (RapidOCR PP-OCRv6 + RapidFuzz + medicines.db)")
    print("=" * 95)

    for img_path in sorted(test_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        img = cv2.imread(str(img_path))
        res = scanner.scan_image(img)
        print(f"\n[Image: {img_path.name}] -> Detected {res.total_strips_detected} strip(s):")
        for s in res.strips:
            m = s.matched_medicine
            brand = m.brand_name if m else "UNMATCHED"
            generic = m.generic if m else "N/A"
            mfg = m.manufacturer if m else "N/A"
            summary = (m.beginner_summary or "")[:90] if m else ""
            print(
                f"  * Strip #{s.strip_index}: Brand={brand:<10} | Strength={str(s.extracted_strength):<32} "
                f"| Form={str(s.detected_dosage_form):<8} | Generic={generic} ({mfg}) | Score={s.match_score}%"
            )
            if summary:
                print(f"    Summary: {summary}...")

    print("\n" + "=" * 95)


if __name__ == "__main__":
    run_benchmark()
