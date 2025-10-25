import { db } from '@/lib/db'
import { mlTrainingData } from '@/lib/db/schema'
import { eq, inArray } from 'drizzle-orm'
import { ensureTrainingDataDir } from '@/lib/storage'
import { writeFile } from 'fs/promises'
import { join } from 'path'

export type TrainingExportResult = {
  count: number
  batches: number
  directory: string
  files: string[]
  exportedIds: string[]
}

export async function exportPendingTrainingData(): Promise<TrainingExportResult> {
  const trainingData = await db
    .select({
      id: mlTrainingData.id,
      gedcomFileId: mlTrainingData.gedcomFileId,
      graphData: mlTrainingData.graphData,
      labels: mlTrainingData.labels,
      exportedAt: mlTrainingData.exportedAt,
      includedInTraining: mlTrainingData.includedInTraining
    })
    .from(mlTrainingData)
    .where(eq(mlTrainingData.includedInTraining, false))

  const trainingDir = await ensureTrainingDataDir()

  if (trainingData.length === 0) {
    return {
      count: 0,
      batches: 0,
      directory: trainingDir,
      files: [],
      exportedIds: []
    }
  }

  const batchSize = 100
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
  const files: string[] = []

  for (let i = 0; i < trainingData.length; i += batchSize) {
    const batch = trainingData.slice(i, i + batchSize)
    const batchNumber = Math.floor(i / batchSize) + 1

    const exportData = {
      metadata: {
        exportedAt: new Date().toISOString(),
        batchNumber,
        totalBatches: Math.ceil(trainingData.length / batchSize),
        count: batch.length
      },
      data: batch.map(item => ({
        id: item.id,
        gedcomFileId: item.gedcomFileId,
        graphData: item.graphData,
        labels: item.labels
      }))
    }

    const filename = `training_batch_${batchNumber}_${timestamp}.json`
    const filepath = join(trainingDir, filename)

    await writeFile(filepath, JSON.stringify(exportData, null, 2))
    files.push(filepath)
  }

  const exportedIds = trainingData.map(item => item.id)
  const now = new Date()

  if (exportedIds.length) {
    await db
      .update(mlTrainingData)
      .set({ exportedAt: now })
      .where(inArray(mlTrainingData.id, exportedIds))
  }

  return {
    count: trainingData.length,
    batches: Math.ceil(trainingData.length / batchSize),
    directory: trainingDir,
    files,
    exportedIds
  }
}

export function generateTrainingScript(dataCount: number): string {
  return `#!/usr/bin/env bash
# Auto-generated on ${new Date().toISOString()}
echo "Training on ${dataCount} samples"
pip install -r ../requirements_mlx.txt
python3 ../train_model_mlx.py --data-dir . --output-dir ../models/parent_predictor
`
}

export function generateRequirements(): string {
  return `mlx
mlx-data
onnx
numpy
scikit-learn
`
}
