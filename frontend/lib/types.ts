export interface BoundingBox {
  id: string;
  label: string;
  brandName: string;
  genericName: string;
  strength: string;
  manufacturer: string;
  confidence: number;
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
  audioUrl?: string;
}

export interface PresetScenario {
  id: string;
  name: string;
  badge: string;
  description: string;
  result: ScanAnalysisResult;
}
