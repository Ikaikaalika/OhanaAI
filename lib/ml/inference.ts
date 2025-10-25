import { existsSync } from 'fs'
import { join } from 'path'
import { FamilyTree } from '@/lib/db/schema'
import { extractPersonFeatures, findSiblings } from '@/lib/ml/dataProcessor'
import * as ort from 'onnxruntime-node'

const MISSING_THRESHOLD = Number(process.env.MISSING_PARENT_THRESHOLD ?? '0.5')
const CONFIDENCE_THRESHOLD = Number(process.env.PREDICTION_CONFIDENCE_THRESHOLD ?? '0.4')
const MAX_SUGGESTIONS = Number(process.env.PREDICTION_MAX_SUGGESTIONS ?? '5')

let session: ort.InferenceSession | null = null
let triedLoading = false

export async function loadModel(): Promise<boolean> {
  if (session) {
    return true
  }
  if (triedLoading) {
    return false
  }
  triedLoading = true
  const modelPath = join(process.cwd(), 'models', 'parent_predictor', 'model.onnx')
  if (!existsSync(modelPath)) {
    console.warn('ONNX model not found at', modelPath)
    return false
  }
  try {
    session = await ort.InferenceSession.create(modelPath)
    return true
  } catch (error) {
    console.error('Failed to load ONNX model', error)
    session = null
    return false
  }
}

export async function runInference(familyTree: FamilyTree, targetPersonId: string): Promise<any[]> {
  const individuals = (familyTree.individuals as any[]) || []
  const relationships = (familyTree.relationships as any) || { parentChild: [], spousal: [] }
  const targetPerson = individuals.find((p) => p.id === targetPersonId)
  if (!targetPerson) {
    throw new Error('Target person not found in tree')
  }

  const { hasMissing, missingFather, missingMother } = session
    ? await predictMissingParents(targetPerson)
    : heuristicMissingParents(targetPerson)

  if (hasMissing < MISSING_THRESHOLD) {
    return []
  }

  const componentIds = getComponentNodes(targetPersonId, individuals, relationships, 200)
  const candidates = individuals.filter((p) => componentIds.includes(p.id) && p.id !== targetPersonId)
  const results: any[] = []

  if (missingFather >= MISSING_THRESHOLD) {
    const ranked = rankCandidates(targetPerson, candidates, 'M')
    results.push(...ranked)
  }

  if (missingMother >= MISSING_THRESHOLD) {
    const ranked = rankCandidates(targetPerson, candidates, 'F')
    results.push(...ranked)
  }

  return results
}

async function predictMissingParents(person: any) {
  if (!session) {
    return heuristicMissingParents(person)
  }
  const features = extractPersonFeatures(person)
  const tensor = new ort.Tensor('float32', Float32Array.from(features), [1, features.length])
  const result = await session.run({ features: tensor })
  const output = result.predictions as ort.Tensor
  const data = Array.from(output.data as Float32Array)
  return {
    hasMissing: data[0] ?? 0,
    missingFather: data[1] ?? 0,
    missingMother: data[2] ?? 0
  }
}

function heuristicMissingParents(person: any) {
  return {
    hasMissing: !person.father || !person.mother ? 1 : 0,
    missingFather: !person.father ? 1 : 0,
    missingMother: !person.mother ? 1 : 0
  }
}

function rankCandidates(target: any, candidates: any[], gender: 'M' | 'F') {
  const filtered = candidates.filter((c) => c.gender === gender)
  const targetYear = extractYear(target?.birthDate)
  const targetLast = getLastName(target)

  const scored = filtered
    .map((c) => {
      const year = extractYear(c?.birthDate)
      const last = getLastName(c)
      const surnameScore = targetLast && last && targetLast === last ? 1 : 0
      const ageDiff = year && targetYear ? Math.abs((year - targetYear) - 30) : 30
      const ageScore = 1 - Math.min(ageDiff / 40, 1)
      const placeScore = locationAffinity(target?.birthPlace, c?.birthPlace)
      const confidence = Number((0.5 * surnameScore + 0.35 * ageScore + 0.15 * placeScore).toFixed(3))
      const reasons = [] as string[]
      if (surnameScore) reasons.push('Shared surname')
      if (ageScore > 0.5) reasons.push('Plausible age gap')
      if (placeScore > 0.5) reasons.push('Birthplace proximity')
      if (!reasons.length) reasons.push('Network proximity')
      return {
        relationship: gender === 'M' ? 'father' : 'mother',
        candidateId: c.id,
        name: c.name || 'Unknown',
        confidence,
        reasons
      }
    })
    .sort((a, b) => b.confidence - a.confidence)

  const filteredByConfidence = scored.filter((entry) => entry.confidence >= CONFIDENCE_THRESHOLD)
  if (!Number.isFinite(MAX_SUGGESTIONS) || MAX_SUGGESTIONS <= 0) {
    return filteredByConfidence
  }
  return filteredByConfidence.slice(0, MAX_SUGGESTIONS)
}

function getLastName(person: any) {
  return (person?.lastName || person?.name || '')
    .toString()
    .split(' ')
    .slice(-1)[0]
    .toLowerCase()
}

function extractYear(dateStr?: string) {
  if (!dateStr) return null
  const match = String(dateStr).match(/\b(\d{4})\b/)
  if (!match) return null
  const year = parseInt(match[1], 10)
  if (year < 1500 || year > 2100) return null
  return year
}

function locationAffinity(a?: string, b?: string) {
  if (!a || !b) return 0
  const la = a.toLowerCase()
  const lb = b.toLowerCase()
  if (la === lb) return 1
  const keys = ['usa', 'united states', 'california', 'hawaii', 'utah', 'nevada']
  let overlap = 0
  for (const key of keys) {
    if (la.includes(key) && lb.includes(key)) {
      overlap += 1
    }
  }
  return Math.min(overlap / keys.length, 1)
}

function getComponentNodes(targetId: string, individuals: any[], relationships: any, maxNodes: number) {
  const edges: Record<string, string[]> = {}
  const ensure = (id: string) => {
    if (!edges[id]) edges[id] = []
  }
  const add = (a: string, b: string) => {
    edges[a].push(b)
    edges[b].push(a)
  }

  const parentChild = relationships.parentChild || []
  const spousal = relationships.spousal || []
  for (const person of individuals) ensure(person.id)
  for (const rel of parentChild) {
    ensure(rel.parent)
    ensure(rel.child)
    add(rel.parent, rel.child)
  }
  for (const rel of spousal) {
    ensure(rel.spouse1)
    ensure(rel.spouse2)
    add(rel.spouse1, rel.spouse2)
  }
  const siblingRels = findSiblings(individuals, parentChild)
  for (const rel of siblingRels) {
    ensure(rel.sibling1)
    ensure(rel.sibling2)
    add(rel.sibling1, rel.sibling2)
  }

  const visited = new Set<string>()
  const queue = [] as string[]
  if (!edges[targetId]) return [targetId]
  queue.push(targetId)
  visited.add(targetId)
  while (queue.length && visited.size < maxNodes) {
    const current = queue.shift()!
    for (const neighbor of edges[current] || []) {
      if (!visited.has(neighbor)) {
        visited.add(neighbor)
        if (visited.size >= maxNodes) break
        queue.push(neighbor)
      }
    }
  }
  return Array.from(visited)
}
