/**
 * OhanaAI - Machine Learning Module
 *
 * Exports all ML-related functionality for predicting missing relatives
 * in family trees.
 */

// Feature extraction
export {
  extractGraphFeatures,
  prepareTrainingData,
  FEATURE_DIMENSIONS,
  type GraphFeatures,
  type IndividualFeatures,
  type EdgeFeature,
  type TrainingExample
} from './features'

// Inference
export {
  InferenceEngine,
  getInferenceEngine,
  predictMissingRelatives,
  formatPredictionResults,
  type MissingRelativePrediction,
  type CandidateMatch,
  type PredictionResult,
  type InferenceConfig
} from './inference'

// Data export
export {
  TrainingDataExporter,
  exportGedcomForTraining,
  migrateTrainingData,
  type ExportConfig,
  type ExportResult,
  type BatchMetadata
} from './exporter'
