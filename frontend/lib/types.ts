export interface MedicineDatabaseRecord {
  brand_id: number;
  brand_name: string;
  type: string | null;
  dosage_form: string | null;
  generic: string;
  manufacturer: string | null;
  package_container: string | null;
  drug_class: string | null;
  indication: string | null;
  dosage_description: string | null;
  interaction_description: string | null;
  side_effects_description: string | null;
  pregnancy_and_lactation_description: string | null;
  beginner_summary?: string | null;
}

export interface BoundingBox {
  id: string;
  label: string;
  brandName: string;
  genericName: string;
  strength: string;
  dosageForm?: string;
  manufacturer: string;
  confidence: number;
  ocrConfidence?: number;
  yoloConfidence?: number | null;
  detectionSource?: string;
  // Normalized coordinates (0-100 percentage)
  x: number;
  y: number;
  width: number;
  height: number;
  rotation?: number;
  status: "safe" | "duplicate" | "interaction" | "warning";
  dosageBangla: string;
  dosageEnglish: string;
  timingTag: string;
  databaseRecord?: MedicineDatabaseRecord;
}

export interface SafetyAlert {
  id: string;
  type: "duplicate" | "interaction" | "safe";
  severity: "critical" | "high" | "moderate" | "safe";
  title: string;
  titleBangla: string;
  description: string;
  descriptionBangla: string;
  involvedBrands: string[];
  genericCompound: string;
  clinicalRisk: string;
}

export interface TransparencyStageTiming {
  id: string;
  label: string;
  engine: string;
  durationMs: number;
  color: string;
}

export interface TransparencyEvidenceItem {
  id: string;
  stripIndex: number;
  brandName: string;
  genericName: string;
  strength: string;
  dosageForm: string;
  manufacturer: string;
  statusBadge: "Safe" | "Caution" | "Danger";
  rawOcrText: string;
  ocrLinesCount: number;
  fuzzyMatchScore: number;
  matchMethod: string;
  confidencePct: number;
  yoloConfidencePct?: number | null;
  detectionSource?: string;
  bboxCoords: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    imgWidth: number;
    imgHeight: number;
  };
  cropThumbnail: string;
  preprocessedThumbnail: string;
  banglaSummary?: string;
}

export interface SafetyEdgeItem {
  id: string;
  rule: string;
  status: "Safe" | "Caution" | "Danger";
  sourceBrand: string;
  targetBrand: string;
  compoundOrClass: string;
  explanationEn: string;
  explanationBn: string;
}

export interface DetectorRuntimeInfo {
  modelLabel?: string;
  variant?: string;
  fileName?: string;
  fileSizeMb?: number;
  isRealModel?: boolean;
  status?: string;
  outputFormat?: string;
  effectiveMode?: string;
  requestedMode?: string;
  confThresholdUsed?: number;
  yoloRawDetections?: number;
  yoloMatchedStrips?: number;
  cropRescuedStrips?: number;
  fallbackRecoveredStrips?: number;
}

export interface TransparencyInspectorData {
  sourceLabel: string;
  totalTurnaroundMs: number;
  wallClockMs?: number;
  activeDetector: string;
  detectorInfo?: DetectorRuntimeInfo;
  activeOcrEngine: string;
  activeReasoningMode: string;
  stages: TransparencyStageTiming[];
  evidenceItems: TransparencyEvidenceItem[];
  safetyEdges: SafetyEdgeItem[];
}

export interface BenchmarkSample {
  id: string;
  filename: string;
  title: string;
  titleBn: string;
  stripsCount: number;
  imageUrl: string;
}

export interface ScanAnalysisResult {
  scanId: string;
  timestamp: string;
  inferenceTimeMs: number;
  pipelineMetrics: {
    yoloDetectionMs: number;
    ocrRectificationMs: number;
    fuzzyEntityMatchMs: number;
    slmSafetyReasoningMs: number;
  };
  overallSafety: "critical_risk" | "moderate_warning" | "all_safe";
  detectedStrips: BoundingBox[];
  alerts: SafetyAlert[];
  banglaVoiceScript: string;
  englishTranslation: string;
  audioUrl?: string | null;
  transparencyInspector?: TransparencyInspectorData;
  sampleId?: string;
  sampleTitle?: string;
  sampleTitleBn?: string;
  demoImageUrl?: string;
}

export interface PresetScenario {
  id: string;
  name: string;
  badge: string;
  description: string;
  result: ScanAnalysisResult;
}
