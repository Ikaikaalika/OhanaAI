#!/usr/bin/env npx ts-node
/**
 * Prepare GEDCOM file for ML training
 *
 * Usage: npx ts-node scripts/prepare_training_data.ts path/to/your/file.ged
 */

import * as fs from 'fs'
import * as path from 'path'
import { parseGedcom } from '../lib/gedcom/parser'
import { extractGraphFeatures, prepareTrainingData, FEATURE_DIMENSIONS } from '../lib/ml/features'

const args = process.argv.slice(2)

if (args.length === 0) {
  console.log('Usage: npx ts-node scripts/prepare_training_data.ts <gedcom-file>')
  console.log('')
  console.log('Example:')
  console.log('  npx ts-node scripts/prepare_training_data.ts ./my-family.ged')
  process.exit(1)
}

const gedcomPath = args[0]

if (!fs.existsSync(gedcomPath)) {
  console.error(`Error: File not found: ${gedcomPath}`)
  process.exit(1)
}

console.log(`Processing: ${gedcomPath}`)

// Read and parse GEDCOM
const gedcomText = fs.readFileSync(gedcomPath, 'utf-8')
const parsed = parseGedcom(gedcomText)

console.log(`  Individuals: ${parsed.individuals.size}`)
console.log(`  Families: ${parsed.families.size}`)
console.log(`  Date range: ${parsed.stats.dateRange.earliest || 'unknown'} - ${parsed.stats.dateRange.latest || 'unknown'}`)
console.log(`  Generations: ${parsed.stats.generations}`)

// Extract features
console.log('\nExtracting features...')
const graphFeatures = extractGraphFeatures(parsed)
const trainingData = prepareTrainingData(graphFeatures)

console.log(`  Nodes: ${trainingData.nodeFeatures.length}`)
console.log(`  Edges: ${trainingData.edgeIndex[0].length}`)
console.log(`  Feature dimension: ${FEATURE_DIMENSIONS.TOTAL}`)

// Prepare output
const outputDir = path.join(process.cwd(), 'training_data')
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true })
}

const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
const baseName = path.basename(gedcomPath, path.extname(gedcomPath))
const outputPath = path.join(outputDir, `training_${baseName}_${timestamp}.json`)

const output = {
  metadata: {
    sourceFile: path.basename(gedcomPath),
    exportedAt: new Date().toISOString(),
    individuals: parsed.individuals.size,
    families: parsed.families.size,
    featureDimension: FEATURE_DIMENSIONS.TOTAL
  },
  data: [{
    id: baseName,
    nodeFeatures: trainingData.nodeFeatures,
    edgeIndex: trainingData.edgeIndex,
    edgeFeatures: trainingData.edgeFeatures,
    edgeTypes: trainingData.edgeTypes,
    labels: trainingData.labels,
    nodeIds: trainingData.nodeIds,
    globalFeatures: trainingData.globalFeatures
  }]
}

fs.writeFileSync(outputPath, JSON.stringify(output, null, 2))

console.log(`\nTraining data saved to: ${outputPath}`)
console.log('\nTo train the model, run:')
console.log('  cd scripts/training && ./run_training.sh')
