import { db } from '@/lib/db'
import { FamilyTree, familyTrees, gedcomFiles, mlTrainingData } from '@/lib/db/schema'
import { processGedcomForML } from '@/lib/ml/dataProcessor'
import { loadModel, runInference } from '@/lib/ml/inference'
import { ParsedGedcom } from '@/lib/gedcom/parser'
import { eq } from 'drizzle-orm'

type Job = {
  fileId: string
  parsedData: ParsedGedcom
}

const queue: Job[] = []
let processing = false

export function enqueueGedcomProcessing(job: Job) {
  queue.push(job)
  if (!processing) {
    void runQueue()
  }
}

export function getPendingGedcomJobs() {
  return queue.length + (processing ? 1 : 0)
}

export async function processGedcomImmediately(job: Job) {
  try {
    await processGedcomJob(job)
  } catch (error) {
    console.error('GEDCOM processing failed', { fileId: job.fileId, error })
    await db
      .update(gedcomFiles)
      .set({
        isProcessed: false,
        processedAt: null
      })
      .where(eq(gedcomFiles.id, job.fileId))
    throw error
  }
}

async function runQueue() {
  processing = true
  while (queue.length > 0) {
    const job = queue.shift()!
    try {
      await processGedcomJob(job)
    } catch (error) {
      console.error('GEDCOM processing failed', { fileId: job.fileId, error })
      await db
        .update(gedcomFiles)
        .set({
          isProcessed: false,
          processedAt: null
        })
        .where(eq(gedcomFiles.id, job.fileId))
    }
  }
  processing = false
}

async function processGedcomJob({ fileId, parsedData }: Job) {
  const individuals = parsedData.individuals || []
  const relationships = parsedData.relationships || []

  const familyTreeRecord = await db.transaction(async (tx) => {
    await tx.delete(familyTrees).where(eq(familyTrees.gedcomFileId, fileId))
    const [tree] = await tx.insert(familyTrees).values({
      gedcomFileId: fileId,
      individuals,
      relationships
    }).returning()

    await tx.delete(mlTrainingData).where(eq(mlTrainingData.gedcomFileId, fileId))
    const mlData = processGedcomForML(parsedData)
    if (mlData) {
      await tx.insert(mlTrainingData).values({
        gedcomFileId: fileId,
        graphData: mlData.graphData,
        labels: mlData.labels
      })
    }

    return tree
  })

  const predictionEntries = familyTreeRecord
    ? await generatePredictionsForTree(familyTreeRecord)
    : []

  await db
    .update(gedcomFiles)
    .set({
      isProcessed: true,
      processedAt: new Date(),
      predictions: predictionEntries,
      modelVersion: predictionEntries.length ? 'onnx-v1' : null
    })
    .where(eq(gedcomFiles.id, fileId))
}

type StoredPrediction = {
  personId: string
  predictions: any[]
}

async function generatePredictionsForTree(familyTree: FamilyTree): Promise<StoredPrediction[]> {
  try {
    const modelLoaded = await loadModel()
    if (!modelLoaded) {
      return []
    }
  } catch (error) {
    console.error('Failed to load model for predictions', error)
    return []
  }

  const individuals = (familyTree.individuals as any[]) || []
  const results: StoredPrediction[] = []

  for (const person of individuals) {
    if (!person?.id) continue
    const missingFather = !person.father
    const missingMother = !person.mother
    if (!missingFather && !missingMother) continue

    try {
      const predictions = await runInference(familyTree, person.id)
      if (Array.isArray(predictions) && predictions.length > 0) {
        results.push({
          personId: person.id,
          predictions
        })
      }
    } catch (error) {
      console.error('Inference error for person', person.id, error)
    }
  }

  return results
}
