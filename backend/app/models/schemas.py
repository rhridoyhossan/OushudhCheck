from typing import List, Optional
from pydantic import BaseModel


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
