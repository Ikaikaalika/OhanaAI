import { mkdir, writeFile, unlink } from 'fs/promises'
import { existsSync } from 'fs'
import { join, resolve, isAbsolute } from 'path'

function ensureAbsolutePath(target: string, base: string) {
  if (!target) return base
  return isAbsolute(target) ? target : resolve(base, target)
}

const projectRoot = process.cwd()
const defaultWritableRoot = process.env.VERCEL ? '/tmp/ohana-ai' : projectRoot
const storageRoot = ensureAbsolutePath(process.env.STORAGE_ROOT || '', defaultWritableRoot)

const uploadsDir = ensureAbsolutePath(process.env.UPLOADS_DIR || '', resolve(storageRoot, 'uploads'))
const trainingDataDir = ensureAbsolutePath(
  process.env.TRAINING_DATA_DIR || '',
  resolve(storageRoot, 'training_data')
)
const mlExportsDir = ensureAbsolutePath(
  process.env.ML_EXPORTS_DIR || '',
  resolve(storageRoot, 'exports/ml_training')
)

async function ensureDir(path: string) {
  if (!existsSync(path)) {
    await mkdir(path, { recursive: true })
  }
}

export function getStorageRoot() {
  return storageRoot
}

export function getUploadsDir() {
  return uploadsDir
}

export async function ensureUploadsDir() {
  await ensureDir(uploadsDir)
  return uploadsDir
}

export function getTrainingDataDir() {
  return trainingDataDir
}

export async function ensureTrainingDataDir() {
  await ensureDir(trainingDataDir)
  return trainingDataDir
}

export function getMlExportsDir() {
  return mlExportsDir
}

export async function ensureMlExportsDir() {
  await ensureDir(mlExportsDir)
  return mlExportsDir
}

export function getUploadFilePath(filename: string) {
  return join(uploadsDir, filename)
}

export async function saveUploadFile(filename: string, contents: Buffer) {
  const dir = await ensureUploadsDir()
  const filePath = join(dir, filename)
  await writeFile(filePath, contents)
  return filePath
}

export async function deleteUploadFile(filename: string) {
  try {
    await unlink(getUploadFilePath(filename))
  } catch (error: any) {
    if (error?.code !== 'ENOENT') {
      throw error
    }
  }
}
