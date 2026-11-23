from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# --- Frontend Database Directory Schemas ---

class MedicineItem(BaseModel):
    brand_id: int
    brand_name: str
    type: Optional[str] = None
    dosage_form: Optional[str] = None
    generic: str
    manufacturer: Optional[str] = None
    package_container: Optional[str] = None
    drug_class: Optional[str] = None
    indication: Optional[str] = None
    dosage_description: Optional[str] = None
    interaction_description: Optional[str] = None
    side_effects_description: Optional[str] = None
    pregnancy_and_lactation_description: Optional[str] = None
    beginner_summary: Optional[str] = None


class DatabaseStats(BaseModel):
    totalBrands: int
    totalGenerics: int


class PaginatedMedicinesResponse(BaseModel):
    items: List[MedicineItem]
    total: int
    page: int
    limit: int
    totalPages: int
    stats: DatabaseStats


# --- Medicine Strip Scanner & Strength Extraction Schemas ---

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0


class OCRLine(BaseModel):
    text: str
    confidence: float = 1.0
    box_height: float = 0.0
    bbox: Optional[BoundingBox] = None


class MedicineInfo(BaseModel):
    brand_id: int
    brand_name: str
    generic: str
    dosage_form: Optional[str] = None
    manufacturer: Optional[str] = None
    package_container: Optional[str] = None
    drug_class: Optional[str] = None
    indication: Optional[str] = None
    indication_description: Optional[str] = None
    dosage_description: Optional[str] = None
    interaction_description: Optional[str] = None
    contraindications_description: Optional[str] = None
    side_effects_description: Optional[str] = None
    pregnancy_and_lactation_description: Optional[str] = None
    precautions_description: Optional[str] = None
    storage_conditions_description: Optional[str] = None
    beginner_summary: Optional[str] = None


class StripScanResult(BaseModel):
    strip_index: int
    bbox: Optional[BoundingBox] = None
    raw_ocr_text: str
    ocr_lines: List[OCRLine] = Field(default_factory=list)
    extracted_brand_query: Optional[str] = None
    extracted_strength: Optional[str] = Field(
        default=None,
        description="Normalized dosage strength extracted from strip, e.g. '500 mg', '1000 mg'",
    )
    all_detected_strengths: List[str] = Field(default_factory=list)
    detected_dosage_form: Optional[str] = None
    match_score: float = 0.0
    match_method: Optional[str] = None
    matched_medicine: Optional[MedicineInfo] = None
    alternatives: List[MedicineInfo] = Field(default_factory=list)
    detection_source: str = Field(
        default="ocr_fallback",
        description="'yolo26_direct' | 'yolo26_crop_rescue' | 'yolo26_split' | 'ocr_fallback'",
    )
    yolo_confidence: Optional[float] = None


class SafetyAlert(BaseModel):
    alert_type: str = Field(description="'duplicate_generic' | 'drug_interaction' | 'same_drug_class'")
    severity: str = Field(description="'high' | 'moderate' | 'info'")
    involved_strips: List[int]
    involved_brands: List[str]
    generic_or_class: str
    message_en: str
    message_bn: str


class MultiStripScanResponse(BaseModel):
    total_strips_detected: int
    strips: List[StripScanResult]
    safety_alerts: List[SafetyAlert] = Field(default_factory=list)
    summary_en: str
    summary_bn: str
    detector_info: Optional[Dict[str, Any]] = None


class TextExtractRequest(BaseModel):
    text: str = Field(..., description="Raw OCR text from a medicine strip, e.g. 'BEXIMCO Napa 500 mg Paracetamol Tablet'")


class MultiTextExtractRequest(BaseModel):
    strip_texts: List[str] = Field(
        ...,
        description="List of raw OCR texts from multiple medicine strips to extract and check for duplicates/interactions",
    )


class TTSRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=6000,
        description="Bangla text to synthesize into neural speech (.mp3)",
    )
