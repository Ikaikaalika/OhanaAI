/**
 * Advanced Feature Engineering for Genealogical ML
 *
 * Extracts comprehensive features from GEDCOM data including:
 * - Demographic features
 * - Temporal features (birth/death/marriage patterns)
 * - Geographic features (location embeddings)
 * - Name features (phonetic, ethnic origin estimation)
 * - Graph structural features (centrality, connectivity)
 * - Relationship pattern features
 */

import {
  ParsedGedcom,
  Individual,
  Family,
  DateInfo,
  PlaceInfo,
  Name
} from '@/lib/gedcom/parser'

// ============================================================================
// Feature Vector Dimensions
// ============================================================================

export const FEATURE_DIMENSIONS = {
  DEMOGRAPHIC: 8,          // Gender, age-related, name basics
  TEMPORAL: 16,            // Birth/death patterns, generation estimation
  GEOGRAPHIC: 32,          // Location embeddings
  NAME_EMBEDDING: 64,      // Character-level name embedding
  GRAPH_STRUCTURAL: 16,    // Node centrality, connectivity
  RELATIONSHIP: 24,        // Missing relatives, family structure
  ATTRIBUTE_MASK: 16,      // Which attributes are present/missing
  TOTAL: 176
}

// ============================================================================
// Core Feature Extraction
// ============================================================================

export interface IndividualFeatures {
  id: string
  demographic: number[]
  temporal: number[]
  geographic: number[]
  nameEmbedding: number[]
  graphStructural: number[]
  relationship: number[]
  attributeMask: number[]

  // Combined flat vector for ML
  flat: number[]

  // Metadata for training labels
  metadata: {
    hasFather: boolean
    hasMother: boolean
    hasSpouse: boolean
    numChildren: number
    numSiblings: number
    generation: number
    isRoot: boolean  // No parents in dataset
    isLeaf: boolean  // No children in dataset
  }
}

export interface GraphFeatures {
  nodes: IndividualFeatures[]
  edges: EdgeFeature[]

  // Global graph statistics
  globalStats: {
    numNodes: number
    numEdges: number
    avgDegree: number
    density: number
    numComponents: number
    dateRangeYears: number
    numGenerations: number
  }
}

export interface EdgeFeature {
  source: string
  target: string
  type: 'parent' | 'child' | 'spouse' | 'sibling'
  weight: number
  features: number[]  // Edge-specific features
}

// ============================================================================
// Feature Extractors
// ============================================================================

/**
 * Extract demographic features
 */
function extractDemographicFeatures(individual: Individual): number[] {
  const features: number[] = []

  // Gender (one-hot: unknown, male, female)
  features.push(individual.gender === undefined ? 1 : 0)  // unknown
  features.push(individual.gender === 'M' ? 1 : 0)        // male
  features.push(individual.gender === 'F' ? 1 : 0)        // female

  // Has name
  features.push(individual.names.length > 0 ? 1 : 0)

  // Name complexity (number of name parts)
  const primaryName = individual.primaryName
  const nameComplexity = primaryName ?
    ((primaryName.given ? 1 : 0) + (primaryName.surname ? 1 : 0) +
     (primaryName.suffix ? 1 : 0) + (primaryName.prefix ? 1 : 0)) / 4 : 0
  features.push(nameComplexity)

  // Has multiple names (aka, married names)
  features.push(individual.names.length > 1 ? 1 : 0)

  // Has occupation
  features.push(individual.occupations.length > 0 ? 1 : 0)

  // Has religion recorded
  features.push(individual.religion ? 1 : 0)

  return features
}

/**
 * Extract temporal features
 */
function extractTemporalFeatures(
  individual: Individual,
  globalStats: { earliest?: number; latest?: number }
): number[] {
  const features: number[] = []

  const yearRange = (globalStats.latest || 2024) - (globalStats.earliest || 1700)
  const normalize = (year: number | undefined) => {
    if (!year || !globalStats.earliest) return 0
    return (year - globalStats.earliest) / Math.max(yearRange, 1)
  }

  // Birth year (normalized to dataset range)
  features.push(normalize(individual.estimatedBirthYear))

  // Death year (normalized)
  features.push(normalize(individual.estimatedDeathYear))

  // Has birth date
  features.push(individual.birth?.date?.year ? 1 : 0)

  // Has death date
  features.push(individual.death?.date?.year ? 1 : 0)

  // Birth date precision (has month/day)
  features.push(individual.birth?.date?.month ? 1 : 0)
  features.push(individual.birth?.date?.day ? 1 : 0)

  // Death date precision
  features.push(individual.death?.date?.month ? 1 : 0)
  features.push(individual.death?.date?.day ? 1 : 0)

  // Lifespan (normalized, typical range 0-100 years)
  const lifespan = individual.lifespan || 0
  features.push(Math.min(lifespan / 100, 1.5))  // Allow up to 150 years

  // Is birth date approximate (circa)
  features.push(individual.birth?.date?.circa ? 1 : 0)

  // Is death date approximate
  features.push(individual.death?.date?.circa ? 1 : 0)

  // Era indicators (one-hot encoding of century)
  const birthCentury = individual.estimatedBirthYear
    ? Math.floor(individual.estimatedBirthYear / 100)
    : 0
  features.push(birthCentury >= 17 && birthCentury < 18 ? 1 : 0)  // 1700s
  features.push(birthCentury >= 18 && birthCentury < 19 ? 1 : 0)  // 1800s
  features.push(birthCentury >= 19 && birthCentury < 20 ? 1 : 0)  // 1900s
  features.push(birthCentury >= 20 ? 1 : 0)                        // 2000s+

  // Has marriage date
  features.push(individual.marriages.length > 0 && individual.marriages[0]?.date ? 1 : 0)

  return features
}

/**
 * Extract geographic features with learned embeddings
 */
function extractGeographicFeatures(
  individual: Individual,
  locationIndex: Map<string, number>,
  locationEmbeddings: number[][]
): number[] {
  const embeddingDim = FEATURE_DIMENSIONS.GEOGRAPHIC
  const features = new Array(embeddingDim).fill(0)

  // Collect all locations for this person
  const locations: PlaceInfo[] = []
  if (individual.birth?.place) locations.push(individual.birth.place)
  if (individual.death?.place) locations.push(individual.death.place)
  for (const res of individual.residences) {
    if (res.place) locations.push(res.place)
  }

  if (locations.length === 0) {
    return features
  }

  // Average embeddings of all locations
  let count = 0
  for (const loc of locations) {
    const normalized = loc.normalized || loc.raw
    const idx = locationIndex.get(normalized)
    if (idx !== undefined && locationEmbeddings[idx]) {
      for (let i = 0; i < embeddingDim; i++) {
        features[i] += locationEmbeddings[idx][i] || 0
      }
      count++
    }
  }

  if (count > 0) {
    for (let i = 0; i < embeddingDim; i++) {
      features[i] /= count
    }
  }

  return features
}

/**
 * Create simple location embeddings based on hierarchical structure
 */
function createLocationEmbeddings(
  locations: Set<string>
): { index: Map<string, number>; embeddings: number[][] } {
  const index = new Map<string, number>()
  const embeddings: number[][] = []
  const embeddingDim = FEATURE_DIMENSIONS.GEOGRAPHIC

  // Parse locations into hierarchies
  const locationList = Array.from(locations)
  const hierarchies: string[][] = locationList.map(loc =>
    loc.split(',').map(s => s.trim()).filter(Boolean)
  )

  // Build vocabulary of location components
  const componentVocab = new Map<string, number>()
  let vocabIdx = 0
  for (const hier of hierarchies) {
    for (const comp of hier) {
      const lower = comp.toLowerCase()
      if (!componentVocab.has(lower)) {
        componentVocab.set(lower, vocabIdx++)
      }
    }
  }

  // Create embeddings using component co-occurrence
  for (let i = 0; i < locationList.length; i++) {
    const loc = locationList[i]
    index.set(loc, i)

    const embedding = new Array(embeddingDim).fill(0)
    const hier = hierarchies[i]

    // Position-based encoding for hierarchy levels
    for (let level = 0; level < hier.length && level < 4; level++) {
      const comp = hier[level].toLowerCase()
      const vocabId = componentVocab.get(comp) || 0

      // Hash component to embedding dimensions
      const startDim = level * 8
      for (let d = 0; d < 8; d++) {
        // Simple hash-based embedding
        const hashVal = ((vocabId * 31 + d) % 1000) / 1000
        embedding[startDim + d] = hashVal * (1 - level * 0.2)  // Decay for lower levels
      }
    }

    embeddings.push(embedding)
  }

  return { index, embeddings }
}

/**
 * Extract character-level name embedding
 */
function extractNameEmbedding(individual: Individual): number[] {
  const embeddingDim = FEATURE_DIMENSIONS.NAME_EMBEDDING
  const features = new Array(embeddingDim).fill(0)

  const name = individual.primaryName
  if (!name) return features

  // Process given name
  const given = (name.given || '').toLowerCase()
  const surname = (name.surname || '').toLowerCase()

  // Character frequency features for given name (first 32 dims)
  for (let i = 0; i < given.length; i++) {
    const code = given.charCodeAt(i) - 97  // 'a' = 0
    if (code >= 0 && code < 26) {
      features[code] += 1 / Math.max(given.length, 1)
    }
  }

  // Character frequency for surname (next 26 dims)
  for (let i = 0; i < surname.length; i++) {
    const code = surname.charCodeAt(i) - 97
    if (code >= 0 && code < 26) {
      features[32 + code] += 1 / Math.max(surname.length, 1)
    }
  }

  // Bigram features (simplified)
  const fullName = `${given} ${surname}`.toLowerCase()
  const commonBigrams = ['th', 'he', 'in', 'er', 'an']
  for (let i = 0; i < commonBigrams.length; i++) {
    features[58 + i] = fullName.includes(commonBigrams[i]) ? 1 : 0
  }

  // Name length features (normalized)
  features[63] = Math.min(given.length / 15, 1)

  return features
}

/**
 * Extract graph structural features
 */
function extractGraphStructuralFeatures(
  individual: Individual,
  individuals: Map<string, Individual>,
  relationships: ParsedGedcom['relationships']
): number[] {
  const features: number[] = []

  // Degree centrality
  const totalConnections =
    (individual.father ? 1 : 0) +
    (individual.mother ? 1 : 0) +
    individual.spouses.length +
    individual.children.length +
    individual.siblings.length

  features.push(Math.min(totalConnections / 20, 1))  // Normalized degree

  // In-degree (people pointing to this person as parent)
  features.push(Math.min(individual.children.length / 10, 1))

  // Out-degree (parents)
  features.push((individual.father ? 0.5 : 0) + (individual.mother ? 0.5 : 0))

  // Sibling count (normalized)
  features.push(Math.min(individual.siblings.length / 10, 1))

  // Spouse count
  features.push(Math.min(individual.spouses.length / 5, 1))

  // Is root (no parents)
  features.push(!individual.father && !individual.mother ? 1 : 0)

  // Is leaf (no children)
  features.push(individual.children.length === 0 ? 1 : 0)

  // Has both parents
  features.push(individual.father && individual.mother ? 1 : 0)

  // Generation estimate (distance from roots)
  const generation = estimateGeneration(individual, individuals)
  features.push(Math.min(generation / 10, 1))

  // Ancestral completeness (how many ancestors are known within 3 generations)
  const ancestralCompleteness = computeAncestralCompleteness(individual, individuals, 3)
  features.push(ancestralCompleteness)

  // Descendant completeness
  const descendantCompleteness = computeDescendantCompleteness(individual, individuals, 3)
  features.push(descendantCompleteness)

  // Local clustering coefficient (simplified)
  const clustering = computeLocalClustering(individual, individuals)
  features.push(clustering)

  // Component size (number of connected individuals)
  const componentSize = computeComponentSize(individual.id, individuals)
  features.push(Math.min(componentSize / individuals.size, 1))

  // Padding
  while (features.length < FEATURE_DIMENSIONS.GRAPH_STRUCTURAL) {
    features.push(0)
  }

  return features.slice(0, FEATURE_DIMENSIONS.GRAPH_STRUCTURAL)
}

/**
 * Extract relationship pattern features
 */
function extractRelationshipFeatures(individual: Individual): number[] {
  const features: number[] = []

  // Missing parent indicators
  features.push(individual.father ? 0 : 1)  // Missing father
  features.push(individual.mother ? 0 : 1)  // Missing mother
  features.push(!individual.father && !individual.mother ? 1 : 0)  // Missing both
  features.push(individual.father && individual.mother ? 1 : 0)  // Has both

  // Spouse status
  features.push(individual.spouses.length === 0 ? 1 : 0)  // No spouse
  features.push(individual.spouses.length === 1 ? 1 : 0)  // One spouse
  features.push(individual.spouses.length > 1 ? 1 : 0)    // Multiple spouses

  // Children status
  features.push(individual.children.length === 0 ? 1 : 0)  // No children
  features.push(individual.children.length > 0 && individual.children.length <= 3 ? 1 : 0)  // 1-3
  features.push(individual.children.length > 3 ? 1 : 0)    // Many children

  // Sibling status
  features.push(individual.siblings.length === 0 ? 1 : 0)  // Only child
  features.push(individual.siblings.length > 0 ? 1 : 0)    // Has siblings

  // Family completeness score
  const completeness = (
    (individual.father ? 0.25 : 0) +
    (individual.mother ? 0.25 : 0) +
    (individual.spouses.length > 0 ? 0.25 : 0) +
    (individual.children.length > 0 || individual.siblings.length > 0 ? 0.25 : 0)
  )
  features.push(completeness)

  // Numerical counts (normalized)
  features.push(Math.min(individual.spouses.length / 5, 1))
  features.push(Math.min(individual.children.length / 15, 1))
  features.push(Math.min(individual.siblings.length / 15, 1))

  // Has grandparents (through parents)
  features.push(0)  // Placeholder - would need parent lookup

  // Has grandchildren
  features.push(0)  // Placeholder

  // Padding
  while (features.length < FEATURE_DIMENSIONS.RELATIONSHIP) {
    features.push(0)
  }

  return features.slice(0, FEATURE_DIMENSIONS.RELATIONSHIP)
}

/**
 * Extract attribute presence mask
 */
function extractAttributeMask(individual: Individual): number[] {
  return [
    individual.names.length > 0 ? 1 : 0,
    individual.primaryName?.given ? 1 : 0,
    individual.primaryName?.surname ? 1 : 0,
    individual.gender !== undefined ? 1 : 0,
    individual.birth?.date ? 1 : 0,
    individual.birth?.place ? 1 : 0,
    individual.death?.date ? 1 : 0,
    individual.death?.place ? 1 : 0,
    individual.residences.length > 0 ? 1 : 0,
    individual.occupations.length > 0 ? 1 : 0,
    individual.religion ? 1 : 0,
    individual.sources.length > 0 ? 1 : 0,
    individual.notes.length > 0 ? 1 : 0,
    individual.marriages.length > 0 ? 1 : 0,
    individual.census.length > 0 ? 1 : 0,
    individual.military.length > 0 ? 1 : 0
  ]
}

// ============================================================================
// Helper Functions
// ============================================================================

function estimateGeneration(
  individual: Individual,
  individuals: Map<string, Individual>,
  visited: Set<string> = new Set()
): number {
  if (visited.has(individual.id)) return 0
  visited.add(individual.id)

  let maxParentGen = 0
  if (individual.father) {
    const father = individuals.get(individual.father)
    if (father) {
      maxParentGen = Math.max(maxParentGen, estimateGeneration(father, individuals, visited) + 1)
    }
  }
  if (individual.mother) {
    const mother = individuals.get(individual.mother)
    if (mother) {
      maxParentGen = Math.max(maxParentGen, estimateGeneration(mother, individuals, visited) + 1)
    }
  }

  return maxParentGen
}

function computeAncestralCompleteness(
  individual: Individual,
  individuals: Map<string, Individual>,
  depth: number
): number {
  if (depth === 0) return 1

  let score = 0
  let maxScore = 2  // Two parents

  if (individual.father) {
    score += 1
    const father = individuals.get(individual.father)
    if (father && depth > 1) {
      score += computeAncestralCompleteness(father, individuals, depth - 1) * 2
      maxScore += 2
    }
  }
  if (individual.mother) {
    score += 1
    const mother = individuals.get(individual.mother)
    if (mother && depth > 1) {
      score += computeAncestralCompleteness(mother, individuals, depth - 1) * 2
      maxScore += 2
    }
  }

  return score / maxScore
}

function computeDescendantCompleteness(
  individual: Individual,
  individuals: Map<string, Individual>,
  depth: number
): number {
  if (depth === 0 || individual.children.length === 0) return 1

  let totalScore = 0
  for (const childId of individual.children) {
    const child = individuals.get(childId)
    if (child) {
      totalScore += computeDescendantCompleteness(child, individuals, depth - 1)
    }
  }

  return totalScore / Math.max(individual.children.length, 1)
}

function computeLocalClustering(
  individual: Individual,
  individuals: Map<string, Individual>
): number {
  // Get all neighbors
  const neighbors = new Set<string>()
  if (individual.father) neighbors.add(individual.father)
  if (individual.mother) neighbors.add(individual.mother)
  individual.spouses.forEach(s => neighbors.add(s))
  individual.children.forEach(c => neighbors.add(c))
  individual.siblings.forEach(s => neighbors.add(s))

  if (neighbors.size < 2) return 0

  // Count edges between neighbors
  let edges = 0
  const neighborList = Array.from(neighbors)
  for (let i = 0; i < neighborList.length; i++) {
    const n1 = individuals.get(neighborList[i])
    if (!n1) continue
    for (let j = i + 1; j < neighborList.length; j++) {
      const n2Id = neighborList[j]
      // Check if n1 and n2 are connected
      if (n1.father === n2Id || n1.mother === n2Id ||
          n1.spouses.includes(n2Id) || n1.children.includes(n2Id) ||
          n1.siblings.includes(n2Id)) {
        edges++
      }
    }
  }

  const maxEdges = (neighbors.size * (neighbors.size - 1)) / 2
  return maxEdges > 0 ? edges / maxEdges : 0
}

function computeComponentSize(
  startId: string,
  individuals: Map<string, Individual>
): number {
  const visited = new Set<string>()
  const queue = [startId]

  while (queue.length > 0) {
    const id = queue.shift()!
    if (visited.has(id)) continue
    visited.add(id)

    const ind = individuals.get(id)
    if (!ind) continue

    // Add all connected individuals to queue
    if (ind.father && !visited.has(ind.father)) queue.push(ind.father)
    if (ind.mother && !visited.has(ind.mother)) queue.push(ind.mother)
    ind.spouses.forEach(s => { if (!visited.has(s)) queue.push(s) })
    ind.children.forEach(c => { if (!visited.has(c)) queue.push(c) })
    ind.siblings.forEach(s => { if (!visited.has(s)) queue.push(s) })
  }

  return visited.size
}

// ============================================================================
// Main Feature Extraction Function
// ============================================================================

export function extractGraphFeatures(parsedGedcom: ParsedGedcom): GraphFeatures {
  const { individuals, relationships, stats } = parsedGedcom

  // Create location embeddings
  const { index: locationIndex, embeddings: locationEmbeddings } =
    createLocationEmbeddings(stats.locations)

  // Extract features for each individual
  const nodes: IndividualFeatures[] = []

  for (const [id, individual] of individuals) {
    const demographic = extractDemographicFeatures(individual)
    const temporal = extractTemporalFeatures(individual, stats.dateRange)
    const geographic = extractGeographicFeatures(individual, locationIndex, locationEmbeddings)
    const nameEmbedding = extractNameEmbedding(individual)
    const graphStructural = extractGraphStructuralFeatures(individual, individuals, relationships)
    const relationship = extractRelationshipFeatures(individual)
    const attributeMask = extractAttributeMask(individual)

    // Combine all features into flat vector
    const flat = [
      ...demographic,
      ...temporal,
      ...geographic,
      ...nameEmbedding,
      ...graphStructural,
      ...relationship,
      ...attributeMask
    ]

    const generation = estimateGeneration(individual, individuals)

    nodes.push({
      id,
      demographic,
      temporal,
      geographic,
      nameEmbedding,
      graphStructural,
      relationship,
      attributeMask,
      flat,
      metadata: {
        hasFather: !!individual.father,
        hasMother: !!individual.mother,
        hasSpouse: individual.spouses.length > 0,
        numChildren: individual.children.length,
        numSiblings: individual.siblings.length,
        generation,
        isRoot: !individual.father && !individual.mother,
        isLeaf: individual.children.length === 0
      }
    })
  }

  // Create edge features
  const edges: EdgeFeature[] = []

  // Parent-child edges
  for (const rel of relationships.parentChild) {
    edges.push({
      source: rel.parent,
      target: rel.child,
      type: 'parent',
      weight: 1.0,
      features: createEdgeFeatures(rel.parent, rel.child, 'parent', individuals)
    })
    // Also add reverse direction
    edges.push({
      source: rel.child,
      target: rel.parent,
      type: 'child',
      weight: 1.0,
      features: createEdgeFeatures(rel.child, rel.parent, 'child', individuals)
    })
  }

  // Spousal edges
  for (const rel of relationships.spousal) {
    edges.push({
      source: rel.spouse1,
      target: rel.spouse2,
      type: 'spouse',
      weight: 1.0,
      features: createEdgeFeatures(rel.spouse1, rel.spouse2, 'spouse', individuals)
    })
    edges.push({
      source: rel.spouse2,
      target: rel.spouse1,
      type: 'spouse',
      weight: 1.0,
      features: createEdgeFeatures(rel.spouse2, rel.spouse1, 'spouse', individuals)
    })
  }

  // Sibling edges
  for (const rel of relationships.sibling) {
    edges.push({
      source: rel.sibling1,
      target: rel.sibling2,
      type: 'sibling',
      weight: 0.8,
      features: createEdgeFeatures(rel.sibling1, rel.sibling2, 'sibling', individuals)
    })
    edges.push({
      source: rel.sibling2,
      target: rel.sibling1,
      type: 'sibling',
      weight: 0.8,
      features: createEdgeFeatures(rel.sibling2, rel.sibling1, 'sibling', individuals)
    })
  }

  // Compute global statistics
  const degrees = nodes.map(n =>
    (n.metadata.hasFather ? 1 : 0) + (n.metadata.hasMother ? 1 : 0) +
    n.metadata.numChildren + (n.metadata.hasSpouse ? 1 : 0) + n.metadata.numSiblings
  )
  const avgDegree = degrees.reduce((a, b) => a + b, 0) / Math.max(degrees.length, 1)
  const maxPossibleEdges = (nodes.length * (nodes.length - 1)) / 2
  const density = maxPossibleEdges > 0 ? edges.length / 2 / maxPossibleEdges : 0

  return {
    nodes,
    edges,
    globalStats: {
      numNodes: nodes.length,
      numEdges: edges.length / 2,  // Divide by 2 because edges are bidirectional
      avgDegree,
      density,
      numComponents: countComponents(individuals),
      dateRangeYears: (stats.dateRange.latest || 2024) - (stats.dateRange.earliest || 1700),
      numGenerations: stats.generations
    }
  }
}

function createEdgeFeatures(
  sourceId: string,
  targetId: string,
  type: 'parent' | 'child' | 'spouse' | 'sibling',
  individuals: Map<string, Individual>
): number[] {
  const features: number[] = []

  const source = individuals.get(sourceId)
  const target = individuals.get(targetId)

  if (!source || !target) {
    return new Array(8).fill(0)
  }

  // Edge type one-hot
  features.push(type === 'parent' ? 1 : 0)
  features.push(type === 'child' ? 1 : 0)
  features.push(type === 'spouse' ? 1 : 0)
  features.push(type === 'sibling' ? 1 : 0)

  // Age difference (normalized)
  const ageDiff = (source.estimatedBirthYear && target.estimatedBirthYear)
    ? (source.estimatedBirthYear - target.estimatedBirthYear) / 100
    : 0
  features.push(ageDiff)

  // Same surname
  const sameSurname = source.primaryName?.surname && target.primaryName?.surname &&
    source.primaryName.surname.toLowerCase() === target.primaryName.surname.toLowerCase()
  features.push(sameSurname ? 1 : 0)

  // Same birthplace
  const sameBirthplace = source.birth?.place?.normalized && target.birth?.place?.normalized &&
    source.birth.place.normalized === target.birth.place.normalized
  features.push(sameBirthplace ? 1 : 0)

  // Both have known dates
  features.push(source.estimatedBirthYear && target.estimatedBirthYear ? 1 : 0)

  return features
}

function countComponents(individuals: Map<string, Individual>): number {
  const visited = new Set<string>()
  let components = 0

  for (const [id] of individuals) {
    if (!visited.has(id)) {
      components++
      // BFS to mark all connected nodes
      const queue = [id]
      while (queue.length > 0) {
        const current = queue.shift()!
        if (visited.has(current)) continue
        visited.add(current)

        const ind = individuals.get(current)
        if (!ind) continue

        if (ind.father && !visited.has(ind.father)) queue.push(ind.father)
        if (ind.mother && !visited.has(ind.mother)) queue.push(ind.mother)
        ind.spouses.forEach(s => { if (!visited.has(s)) queue.push(s) })
        ind.children.forEach(c => { if (!visited.has(c)) queue.push(c) })
        ind.siblings.forEach(s => { if (!visited.has(s)) queue.push(s) })
      }
    }
  }

  return components
}

// ============================================================================
// Training Data Export
// ============================================================================

export interface TrainingExample {
  nodeFeatures: number[][]  // [numNodes, featureDim]
  edgeIndex: number[][]     // [2, numEdges] - source and target indices
  edgeFeatures: number[][]  // [numEdges, edgeFeatureDim]
  edgeTypes: number[]       // [numEdges] - 0=parent, 1=child, 2=spouse, 3=sibling

  // Labels for multi-task learning
  labels: {
    missingFather: number[]    // [numNodes] - binary
    missingMother: number[]    // [numNodes] - binary
    missingSpouse: number[]    // [numNodes] - binary
    missingChildren: number[]  // [numNodes] - binary (has children but might have more)
    missingSiblings: number[]  // [numNodes] - binary
  }

  // Node ID mapping
  nodeIds: string[]

  // Global graph features
  globalFeatures: number[]
}

export function prepareTrainingData(graphFeatures: GraphFeatures): TrainingExample {
  const { nodes, edges, globalStats } = graphFeatures

  // Create node ID to index mapping
  const nodeIdToIndex = new Map<string, number>()
  nodes.forEach((node, index) => {
    nodeIdToIndex.set(node.id, index)
  })

  // Prepare node features matrix
  const nodeFeatures = nodes.map(n => n.flat)

  // Prepare edge index and features
  const edgeIndex: [number[], number[]] = [[], []]
  const edgeFeatures: number[][] = []
  const edgeTypes: number[] = []

  const typeToIndex: Record<string, number> = {
    'parent': 0,
    'child': 1,
    'spouse': 2,
    'sibling': 3
  }

  for (const edge of edges) {
    const sourceIdx = nodeIdToIndex.get(edge.source)
    const targetIdx = nodeIdToIndex.get(edge.target)

    if (sourceIdx !== undefined && targetIdx !== undefined) {
      edgeIndex[0].push(sourceIdx)
      edgeIndex[1].push(targetIdx)
      edgeFeatures.push(edge.features)
      edgeTypes.push(typeToIndex[edge.type] || 0)
    }
  }

  // Prepare labels
  const labels = {
    missingFather: nodes.map(n => n.metadata.hasFather ? 0 : 1),
    missingMother: nodes.map(n => n.metadata.hasMother ? 0 : 1),
    missingSpouse: nodes.map(n => n.metadata.hasSpouse ? 0 : 1),
    missingChildren: nodes.map(n => n.metadata.numChildren === 0 ? 1 : 0),
    missingSiblings: nodes.map(n => n.metadata.numSiblings === 0 ? 1 : 0)
  }

  // Global features
  const globalFeatures = [
    Math.min(globalStats.numNodes / 10000, 1),
    Math.min(globalStats.numEdges / 50000, 1),
    Math.min(globalStats.avgDegree / 10, 1),
    globalStats.density,
    Math.min(globalStats.numComponents / 100, 1),
    Math.min(globalStats.dateRangeYears / 500, 1),
    Math.min(globalStats.numGenerations / 15, 1)
  ]

  return {
    nodeFeatures,
    edgeIndex,
    edgeFeatures,
    edgeTypes,
    labels,
    nodeIds: nodes.map(n => n.id),
    globalFeatures
  }
}
