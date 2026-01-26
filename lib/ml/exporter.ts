/**
 * OhanaAI - Training Data Exporter
 *
 * Exports processed GEDCOM data in formats suitable for ML training:
 * 1. JSON format for the MLX training pipeline
 * 2. Batch processing for large datasets
 * 3. Train/validation split support
 */

import { ParsedGedcom } from '@/lib/gedcom/parser'
import {
  extractGraphFeatures,
  prepareTrainingData,
  TrainingExample,
  FEATURE_DIMENSIONS
} from './features'
import * as fs from 'fs'
import * as path from 'path'

// ============================================================================
// Types
// ============================================================================

export interface ExportConfig {
  outputDir: string
  batchSize: number
  validationSplit: number
  includeRawData: boolean
  format: 'json' | 'jsonl'
}

export interface ExportResult {
  trainFiles: string[]
  valFiles: string[]
  totalExamples: number
  trainExamples: number
  valExamples: number
  featureDimension: number
  timestamp: Date
}

export interface BatchMetadata {
  batchNumber: number
  totalBatches: number
  exampleCount: number
  exportedAt: string
  featureDimension: number
  labelTypes: string[]
  split: 'train' | 'validation'
}

// ============================================================================
// Exporter Class
// ============================================================================

export class TrainingDataExporter {
  private config: ExportConfig

  constructor(config?: Partial<ExportConfig>) {
    this.config = {
      outputDir: config?.outputDir || 'training_data',
      batchSize: config?.batchSize || 1000,
      validationSplit: config?.validationSplit || 0.2,
      includeRawData: config?.includeRawData ?? false,
      format: config?.format || 'json'
    }
  }

  /**
   * Export a single GEDCOM file to training format
   */
  exportSingle(
    parsedGedcom: ParsedGedcom,
    fileId: string
  ): TrainingExample {
    const graphFeatures = extractGraphFeatures(parsedGedcom)
    const trainingData = prepareTrainingData(graphFeatures)

    return trainingData
  }

  /**
   * Export multiple GEDCOM files as batched training data
   */
  async exportBatched(
    gedcomFiles: Array<{ id: string; parsed: ParsedGedcom }>,
    prefix: string = 'training_batch'
  ): Promise<ExportResult> {
    // Ensure output directory exists
    if (!fs.existsSync(this.config.outputDir)) {
      fs.mkdirSync(this.config.outputDir, { recursive: true })
    }

    // Process all files
    const allExamples: Array<{ id: string; data: TrainingExample }> = []

    for (const file of gedcomFiles) {
      try {
        const example = this.exportSingle(file.parsed, file.id)
        allExamples.push({ id: file.id, data: example })
      } catch (error) {
        console.error(`Error processing file ${file.id}:`, error)
      }
    }

    if (allExamples.length === 0) {
      throw new Error('No valid training examples generated')
    }

    // Shuffle for train/val split
    const shuffled = this.shuffle(allExamples)
    const splitIdx = Math.floor(shuffled.length * (1 - this.config.validationSplit))
    const trainExamples = shuffled.slice(0, splitIdx)
    const valExamples = shuffled.slice(splitIdx)

    // Export train batches
    const trainFiles = await this.writeBatches(
      trainExamples,
      `${prefix}_train`,
      'train'
    )

    // Export validation batches
    const valFiles = await this.writeBatches(
      valExamples,
      `${prefix}_val`,
      'validation'
    )

    return {
      trainFiles,
      valFiles,
      totalExamples: allExamples.length,
      trainExamples: trainExamples.length,
      valExamples: valExamples.length,
      featureDimension: FEATURE_DIMENSIONS.TOTAL,
      timestamp: new Date()
    }
  }

  /**
   * Export a single training example to the new format
   */
  exportToNewFormat(example: TrainingExample): object {
    return {
      nodeFeatures: example.nodeFeatures,
      edgeIndex: example.edgeIndex,
      edgeFeatures: example.edgeFeatures,
      edgeTypes: example.edgeTypes,
      labels: {
        missingFather: example.labels.missingFather,
        missingMother: example.labels.missingMother,
        missingSpouse: example.labels.missingSpouse,
        missingChildren: example.labels.missingChildren,
        missingSiblings: example.labels.missingSiblings
      },
      nodeIds: example.nodeIds,
      globalFeatures: example.globalFeatures
    }
  }

  /**
   * Convert old format training data to new format
   */
  convertOldFormat(oldData: any): TrainingExample | null {
    try {
      const graphData = oldData.graphData || {}
      const labels = oldData.labels || []
      const nodes = graphData.nodes || []
      const edges = graphData.edges || []

      if (nodes.length < 3) return null

      // Build node features
      const nodeIdToIdx = new Map<string, number>()
      const nodeFeatures: number[][] = []
      const nodeIds: string[] = []

      nodes.forEach((node: any, idx: number) => {
        nodeIdToIdx.set(node.id, idx)
        nodeIds.push(node.id)

        // Pad features to new dimension
        const features = node.features || []
        const padded = [...features]
        while (padded.length < FEATURE_DIMENSIONS.TOTAL) {
          padded.push(0)
        }
        nodeFeatures.push(padded.slice(0, FEATURE_DIMENSIONS.TOTAL))
      })

      // Build edge index
      const edgeIndex: [number[], number[]] = [[], []]
      const edgeFeatures: number[][] = []
      const edgeTypes: number[] = []

      const typeToIdx: Record<string, number> = {
        parent: 0, child: 1, spouse: 2, sibling: 3
      }

      for (const edge of edges) {
        const srcIdx = nodeIdToIdx.get(edge.source)
        const tgtIdx = nodeIdToIdx.get(edge.target)

        if (srcIdx !== undefined && tgtIdx !== undefined) {
          edgeIndex[0].push(srcIdx)
          edgeIndex[1].push(tgtIdx)

          const edgeFeat = new Array(8).fill(0)
          edgeFeat[typeToIdx[edge.type] || 0] = 1
          edgeFeat[4] = edge.weight || 1.0
          edgeFeatures.push(edgeFeat)
          edgeTypes.push(typeToIdx[edge.type] || 0)
        }
      }

      // Build labels
      const labelArrays = {
        missingFather: new Array(nodes.length).fill(0),
        missingMother: new Array(nodes.length).fill(0),
        missingSpouse: new Array(nodes.length).fill(0),
        missingChildren: new Array(nodes.length).fill(0),
        missingSiblings: new Array(nodes.length).fill(0)
      }

      for (const label of labels) {
        const idx = nodeIdToIdx.get(label.personId)
        if (idx === undefined) continue

        const hasMissing = label.hasMissingParent
        const missingType = label.missingParentType

        if (missingType === 'father' || missingType === 'both') {
          labelArrays.missingFather[idx] = 1
        }
        if (missingType === 'mother' || missingType === 'both') {
          labelArrays.missingMother[idx] = 1
        }
        if (hasMissing && !missingType) {
          labelArrays.missingFather[idx] = 1
          labelArrays.missingMother[idx] = 1
        }

        // Infer spouse/children from attributes
        const attrs = label.attributes || {}
        const missingAttrs = label.missingAttributes || {}

        if (!attrs.spouses || attrs.spouses.length === 0) {
          labelArrays.missingSpouse[idx] = 1
        }
      }

      return {
        nodeFeatures,
        edgeIndex,
        edgeFeatures,
        edgeTypes,
        labels: labelArrays,
        nodeIds,
        globalFeatures: new Array(7).fill(0)
      }

    } catch (error) {
      console.error('Error converting old format:', error)
      return null
    }
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private async writeBatches(
    examples: Array<{ id: string; data: TrainingExample }>,
    prefix: string,
    split: 'train' | 'validation'
  ): Promise<string[]> {
    const files: string[] = []
    const totalBatches = Math.ceil(examples.length / this.config.batchSize)

    for (let batch = 0; batch < totalBatches; batch++) {
      const start = batch * this.config.batchSize
      const end = Math.min(start + this.config.batchSize, examples.length)
      const batchExamples = examples.slice(start, end)

      const metadata: BatchMetadata = {
        batchNumber: batch + 1,
        totalBatches,
        exampleCount: batchExamples.length,
        exportedAt: new Date().toISOString(),
        featureDimension: FEATURE_DIMENSIONS.TOTAL,
        labelTypes: ['missingFather', 'missingMother', 'missingSpouse', 'missingChildren', 'missingSiblings'],
        split
      }

      const batchData = {
        metadata,
        data: batchExamples.map(ex => ({
          id: ex.id,
          ...this.exportToNewFormat(ex.data)
        }))
      }

      const filename = `${prefix}_${batch + 1}_${Date.now()}.json`
      const filepath = path.join(this.config.outputDir, filename)

      if (this.config.format === 'jsonl') {
        // Write as JSON lines
        const lines = batchExamples.map(ex =>
          JSON.stringify({ id: ex.id, ...this.exportToNewFormat(ex.data) })
        )
        fs.writeFileSync(filepath.replace('.json', '.jsonl'), lines.join('\n'))
      } else {
        fs.writeFileSync(filepath, JSON.stringify(batchData, null, 2))
      }

      files.push(filepath)
    }

    return files
  }

  private shuffle<T>(array: T[]): T[] {
    const result = [...array]
    for (let i = result.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[result[i], result[j]] = [result[j], result[i]]
    }
    return result
  }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * Quick export of a GEDCOM file to training format
 */
export function exportGedcomForTraining(
  parsedGedcom: ParsedGedcom,
  outputPath: string
): void {
  const exporter = new TrainingDataExporter()
  const example = exporter.exportSingle(parsedGedcom, 'single')
  const formatted = exporter.exportToNewFormat(example)

  fs.writeFileSync(outputPath, JSON.stringify({
    metadata: {
      exportedAt: new Date().toISOString(),
      featureDimension: FEATURE_DIMENSIONS.TOTAL,
      nodeCount: example.nodeFeatures.length,
      edgeCount: example.edgeIndex[0].length
    },
    ...formatted
  }, null, 2))
}

/**
 * Convert existing training data to new format
 */
export function migrateTrainingData(
  inputDir: string,
  outputDir: string
): void {
  const exporter = new TrainingDataExporter({ outputDir })

  const files = fs.readdirSync(inputDir).filter(f => f.endsWith('.json'))

  for (const file of files) {
    try {
      const content = fs.readFileSync(path.join(inputDir, file), 'utf-8')
      const data = JSON.parse(content)

      const examples = data.data || [data]
      const converted: any[] = []

      for (const item of examples) {
        const result = exporter.convertOldFormat(item)
        if (result) {
          converted.push({
            id: item.id || item.gedcomFileId || 'unknown',
            ...exporter.exportToNewFormat(result)
          })
        }
      }

      if (converted.length > 0) {
        const outputPath = path.join(outputDir, `converted_${file}`)
        fs.writeFileSync(outputPath, JSON.stringify({
          metadata: {
            convertedAt: new Date().toISOString(),
            originalFile: file,
            exampleCount: converted.length,
            featureDimension: FEATURE_DIMENSIONS.TOTAL
          },
          data: converted
        }, null, 2))

        console.log(`Converted ${file}: ${converted.length} examples`)
      }
    } catch (error) {
      console.error(`Error converting ${file}:`, error)
    }
  }
}
