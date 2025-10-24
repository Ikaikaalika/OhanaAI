import { db } from '@/lib/db'
import { familyTrees, gedcomFiles, mlTrainingData } from '@/lib/db/schema'
import { processGedcomForML } from '@/lib/ml/dataProcessor'
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

async function runQueue() {
  processing = true
  while (queue.length > 0) {
    const job = queue.shift()!
    try {
      await processJob(job)
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

async function processJob({ fileId, parsedData }: Job) {
  const individuals = parsedData.individuals || []
  const relationships = parsedData.relationships || []

  await db.transaction(async (tx) => {
    await tx.delete(familyTrees).where(eq(familyTrees.gedcomFileId, fileId))
    await tx.insert(familyTrees).values({
      gedcomFileId: fileId,
      individuals,
      relationships
    })

    await tx.delete(mlTrainingData).where(eq(mlTrainingData.gedcomFileId, fileId))
    const mlData = processGedcomForML(parsedData)
    if (mlData) {
      await tx.insert(mlTrainingData).values({
        gedcomFileId: fileId,
        graphData: mlData.graphData,
        labels: mlData.labels
      })
    }
  })

  await db
    .update(gedcomFiles)
    .set({
      isProcessed: true,
      processedAt: new Date()
    })
    .where(eq(gedcomFiles.id, fileId))
}
