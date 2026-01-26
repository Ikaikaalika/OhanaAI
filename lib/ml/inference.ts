/**
 * OhanaAI - Inference System for Missing Relative Prediction
 *
 * This module provides:
 * 1. Model loading and caching
 * 2. Batch inference for missing relative detection
 * 3. Candidate ranking with confidence scores
 * 4. Heuristic fallbacks when model unavailable
 */

import { ParsedGedcom, Individual, getIndividualDisplayName } from '@/lib/gedcom/parser'
import {
  extractGraphFeatures,
  prepareTrainingData,
  GraphFeatures,
  IndividualFeatures
} from './features'

// ============================================================================
// Types
// ============================================================================

export interface MissingRelativePrediction {
  personId: string
  personName: string

  // Probabilities for each relation type
  missingFather: number
  missingMother: number
  missingSpouse: number
  missingChildren: number
  missingSiblings: number

  // Overall missing relative score
  overallMissingScore: number

  // Confidence based on data quality
  confidence: number
}

export interface CandidateMatch {
  candidateId: string
  candidateName: string
  relationshipType: 'father' | 'mother' | 'spouse' | 'sibling' | 'child'
  score: number
  confidence: number

  // Explanation factors
  factors: {
    surnameMatch: number
    ageCompatibility: number
    locationMatch: number
    networkProximity: number
    temporalPlausibility: number
  }

  // Additional context
  candidateBirthYear?: number
  candidateDeathYear?: number
  candidateBirthPlace?: string
}

export interface PredictionResult {
  predictions: MissingRelativePrediction[]
  candidates: Map<string, CandidateMatch[]>  // personId -> candidates
  modelVersion: string
  timestamp: Date
  processingTimeMs: number
}

export interface InferenceConfig {
  modelPath?: string
  confidenceThreshold: number
  maxCandidatesPerPerson: number
  useHeuristics: boolean
  batchSize: number
}

// ============================================================================
// Inference Engine
// ============================================================================

export class InferenceEngine {
  private config: InferenceConfig
  private modelLoaded: boolean = false
  private modelVersion: string = 'heuristic-v1'

  constructor(config?: Partial<InferenceConfig>) {
    this.config = {
      modelPath: config?.modelPath || 'models/family_tree_gnn/best_model.npz',
      confidenceThreshold: config?.confidenceThreshold || 0.4,
      maxCandidatesPerPerson: config?.maxCandidatesPerPerson || 10,
      useHeuristics: config?.useHeuristics ?? true,
      batchSize: config?.batchSize || 100
    }
  }

  /**
   * Run full inference pipeline on a parsed GEDCOM file
   */
  async predict(parsedGedcom: ParsedGedcom): Promise<PredictionResult> {
    const startTime = Date.now()

    // Extract features
    const graphFeatures = extractGraphFeatures(parsedGedcom)

    // Generate predictions for each individual
    const predictions: MissingRelativePrediction[] = []
    const candidates = new Map<string, CandidateMatch[]>()

    for (const nodeFeatures of graphFeatures.nodes) {
      const individual = parsedGedcom.individuals.get(nodeFeatures.id)
      if (!individual) continue

      // Predict missing relatives
      const prediction = this.predictMissingRelatives(individual, nodeFeatures, parsedGedcom)
      predictions.push(prediction)

      // Find candidates for missing relations
      const personCandidates: CandidateMatch[] = []

      if (prediction.missingFather > this.config.confidenceThreshold) {
        const fatherCandidates = this.findCandidates(
          individual, 'father', graphFeatures, parsedGedcom
        )
        personCandidates.push(...fatherCandidates)
      }

      if (prediction.missingMother > this.config.confidenceThreshold) {
        const motherCandidates = this.findCandidates(
          individual, 'mother', graphFeatures, parsedGedcom
        )
        personCandidates.push(...motherCandidates)
      }

      if (prediction.missingSpouse > this.config.confidenceThreshold) {
        const spouseCandidates = this.findCandidates(
          individual, 'spouse', graphFeatures, parsedGedcom
        )
        personCandidates.push(...spouseCandidates)
      }

      if (personCandidates.length > 0) {
        candidates.set(individual.id, personCandidates)
      }
    }

    return {
      predictions,
      candidates,
      modelVersion: this.modelVersion,
      timestamp: new Date(),
      processingTimeMs: Date.now() - startTime
    }
  }

  /**
   * Predict missing relatives for a single individual
   */
  private predictMissingRelatives(
    individual: Individual,
    features: IndividualFeatures,
    parsedGedcom: ParsedGedcom
  ): MissingRelativePrediction {
    // Use extracted metadata for ground truth
    const hasFather = features.metadata.hasFather
    const hasMother = features.metadata.hasMother
    const hasSpouse = features.metadata.hasSpouse

    // Base probabilities from known data
    let missingFather = hasFather ? 0.1 : 0.9
    let missingMother = hasMother ? 0.1 : 0.9
    let missingSpouse = hasSpouse ? 0.2 : 0.7
    let missingChildren = features.metadata.numChildren === 0 ? 0.6 : 0.2
    let missingSiblings = features.metadata.numSiblings === 0 ? 0.7 : 0.3

    // Adjust based on data completeness
    const dataQuality = this.assessDataQuality(features)

    // If we have good data and person is root, they likely have unknown parents
    if (features.metadata.isRoot && dataQuality > 0.5) {
      missingFather = Math.max(missingFather, 0.85)
      missingMother = Math.max(missingMother, 0.85)
    }

    // Adjust spouse probability based on age and era
    if (this.isMarriageAgeRange(individual) && !hasSpouse) {
      missingSpouse = Math.max(missingSpouse, 0.75)
    }

    // Calculate confidence based on available data
    const confidence = this.calculatePredictionConfidence(individual, features)

    // Overall missing score
    const overallMissingScore = (
      missingFather * 0.3 +
      missingMother * 0.3 +
      missingSpouse * 0.2 +
      missingChildren * 0.1 +
      missingSiblings * 0.1
    )

    return {
      personId: individual.id,
      personName: getIndividualDisplayName(individual),
      missingFather,
      missingMother,
      missingSpouse,
      missingChildren,
      missingSiblings,
      overallMissingScore,
      confidence
    }
  }

  /**
   * Find and rank candidate relatives
   */
  private findCandidates(
    individual: Individual,
    relationshipType: 'father' | 'mother' | 'spouse' | 'sibling' | 'child',
    graphFeatures: GraphFeatures,
    parsedGedcom: ParsedGedcom
  ): CandidateMatch[] {
    const candidates: CandidateMatch[] = []

    // Filter candidates by gender and age appropriateness
    const validCandidates = this.filterCandidatesByRelationType(
      individual,
      relationshipType,
      parsedGedcom
    )

    // Score each candidate
    for (const candidateId of validCandidates) {
      const candidate = parsedGedcom.individuals.get(candidateId)
      if (!candidate) continue

      const factors = this.calculateMatchFactors(
        individual, candidate, relationshipType, parsedGedcom
      )

      // Combined score
      const score = this.calculateCombinedScore(factors, relationshipType)
      const confidence = this.calculateMatchConfidence(individual, candidate, factors)

      if (score >= this.config.confidenceThreshold) {
        candidates.push({
          candidateId: candidate.id,
          candidateName: getIndividualDisplayName(candidate),
          relationshipType,
          score,
          confidence,
          factors,
          candidateBirthYear: candidate.estimatedBirthYear,
          candidateDeathYear: candidate.estimatedDeathYear,
          candidateBirthPlace: candidate.birth?.place?.normalized
        })
      }
    }

    // Sort by score and limit
    candidates.sort((a, b) => b.score - a.score)
    return candidates.slice(0, this.config.maxCandidatesPerPerson)
  }

  /**
   * Filter candidates based on relationship type constraints
   */
  private filterCandidatesByRelationType(
    individual: Individual,
    relationshipType: 'father' | 'mother' | 'spouse' | 'sibling' | 'child',
    parsedGedcom: ParsedGedcom
  ): string[] {
    const validCandidates: string[] = []
    const individualBirthYear = individual.estimatedBirthYear

    for (const [id, candidate] of parsedGedcom.individuals) {
      // Skip self
      if (id === individual.id) continue

      // Skip already known relatives
      if (individual.father === id || individual.mother === id) continue
      if (individual.spouses.includes(id)) continue
      if (individual.children.includes(id)) continue
      if (individual.siblings.includes(id)) continue

      const candidateBirthYear = candidate.estimatedBirthYear

      switch (relationshipType) {
        case 'father':
          // Must be male and older
          if (candidate.gender !== 'M') continue
          if (candidateBirthYear && individualBirthYear) {
            const ageDiff = individualBirthYear - candidateBirthYear
            if (ageDiff < 15 || ageDiff > 70) continue  // Reasonable parent age
          }
          break

        case 'mother':
          // Must be female and older
          if (candidate.gender !== 'F') continue
          if (candidateBirthYear && individualBirthYear) {
            const ageDiff = individualBirthYear - candidateBirthYear
            if (ageDiff < 12 || ageDiff > 55) continue  // Reasonable mother age
          }
          break

        case 'spouse':
          // Should be opposite gender (or allow same for modern records)
          // Age should be similar (within ~20 years typically)
          if (candidateBirthYear && individualBirthYear) {
            const ageDiff = Math.abs(individualBirthYear - candidateBirthYear)
            if (ageDiff > 30) continue
          }
          break

        case 'sibling':
          // Should be similar age (within ~25 years)
          if (candidateBirthYear && individualBirthYear) {
            const ageDiff = Math.abs(individualBirthYear - candidateBirthYear)
            if (ageDiff > 25) continue
          }
          break

        case 'child':
          // Must be younger
          if (candidateBirthYear && individualBirthYear) {
            const ageDiff = candidateBirthYear - individualBirthYear
            if (ageDiff < 12 || ageDiff > 70) continue
          }
          break
      }

      validCandidates.push(id)
    }

    return validCandidates
  }

  /**
   * Calculate matching factors between individual and candidate
   */
  private calculateMatchFactors(
    individual: Individual,
    candidate: Individual,
    relationshipType: string,
    parsedGedcom: ParsedGedcom
  ): CandidateMatch['factors'] {
    // Surname matching
    const surnameMatch = this.calculateSurnameMatch(individual, candidate, relationshipType)

    // Age compatibility
    const ageCompatibility = this.calculateAgeCompatibility(
      individual, candidate, relationshipType
    )

    // Location matching
    const locationMatch = this.calculateLocationMatch(individual, candidate)

    // Network proximity (shared connections)
    const networkProximity = this.calculateNetworkProximity(
      individual, candidate, parsedGedcom
    )

    // Temporal plausibility
    const temporalPlausibility = this.calculateTemporalPlausibility(
      individual, candidate, relationshipType
    )

    return {
      surnameMatch,
      ageCompatibility,
      locationMatch,
      networkProximity,
      temporalPlausibility
    }
  }

  /**
   * Calculate surname match score
   */
  private calculateSurnameMatch(
    individual: Individual,
    candidate: Individual,
    relationshipType: string
  ): number {
    const indSurname = individual.primaryName?.surname?.toLowerCase()
    const candSurname = candidate.primaryName?.surname?.toLowerCase()

    if (!indSurname || !candSurname) return 0.3  // Unknown

    if (indSurname === candSurname) {
      // Exact match - very important for fathers, siblings
      if (relationshipType === 'father' || relationshipType === 'sibling') {
        return 1.0
      }
      return 0.8
    }

    // Check for similar surnames (Soundex or edit distance)
    const similarity = this.calculateStringSimilarity(indSurname, candSurname)
    return similarity * 0.7
  }

  /**
   * Calculate age compatibility score
   */
  private calculateAgeCompatibility(
    individual: Individual,
    candidate: Individual,
    relationshipType: string
  ): number {
    const indYear = individual.estimatedBirthYear
    const candYear = candidate.estimatedBirthYear

    if (!indYear || !candYear) return 0.5  // Unknown

    const ageDiff = indYear - candYear

    switch (relationshipType) {
      case 'father':
        // Optimal: 20-35 years older
        if (ageDiff >= 20 && ageDiff <= 35) return 1.0
        if (ageDiff >= 15 && ageDiff <= 45) return 0.7
        if (ageDiff >= 12 && ageDiff <= 60) return 0.4
        return 0.1

      case 'mother':
        // Optimal: 18-30 years older
        if (ageDiff >= 18 && ageDiff <= 30) return 1.0
        if (ageDiff >= 15 && ageDiff <= 40) return 0.7
        if (ageDiff >= 12 && ageDiff <= 50) return 0.4
        return 0.1

      case 'spouse':
        // Optimal: within 5 years
        const absAgeDiff = Math.abs(ageDiff)
        if (absAgeDiff <= 5) return 1.0
        if (absAgeDiff <= 10) return 0.8
        if (absAgeDiff <= 20) return 0.5
        return 0.2

      case 'sibling':
        // Optimal: within 10 years
        const siblingAgeDiff = Math.abs(ageDiff)
        if (siblingAgeDiff <= 5) return 1.0
        if (siblingAgeDiff <= 10) return 0.8
        if (siblingAgeDiff <= 20) return 0.4
        return 0.1

      default:
        return 0.5
    }
  }

  /**
   * Calculate location match score
   */
  private calculateLocationMatch(
    individual: Individual,
    candidate: Individual
  ): number {
    const indPlaces = this.collectPlaces(individual)
    const candPlaces = this.collectPlaces(candidate)

    if (indPlaces.length === 0 || candPlaces.length === 0) return 0.3

    let maxMatch = 0
    for (const indPlace of indPlaces) {
      for (const candPlace of candPlaces) {
        const match = this.calculatePlaceMatch(indPlace, candPlace)
        maxMatch = Math.max(maxMatch, match)
      }
    }

    return maxMatch
  }

  /**
   * Calculate network proximity (shared relatives/connections)
   */
  private calculateNetworkProximity(
    individual: Individual,
    candidate: Individual,
    parsedGedcom: ParsedGedcom
  ): number {
    // Check for shared relatives
    let sharedConnections = 0

    // Shared parents
    if (individual.father && individual.father === candidate.father) sharedConnections += 2
    if (individual.mother && individual.mother === candidate.mother) sharedConnections += 2

    // Shared spouses (unlikely but possible step-family)
    for (const spouse of individual.spouses) {
      if (candidate.spouses.includes(spouse)) sharedConnections += 1
    }

    // Shared children
    for (const child of individual.children) {
      if (candidate.children.includes(child)) sharedConnections += 1.5
    }

    // Candidate is parent of a sibling
    for (const sibling of individual.siblings) {
      const sib = parsedGedcom.individuals.get(sibling)
      if (sib && (sib.father === candidate.id || sib.mother === candidate.id)) {
        sharedConnections += 3  // Strong signal
      }
    }

    // Normalize
    return Math.min(sharedConnections / 5, 1.0)
  }

  /**
   * Calculate temporal plausibility
   */
  private calculateTemporalPlausibility(
    individual: Individual,
    candidate: Individual,
    relationshipType: string
  ): number {
    // Check if candidate was alive when individual was born
    const indBirthYear = individual.estimatedBirthYear
    const candBirthYear = candidate.estimatedBirthYear
    const candDeathYear = candidate.estimatedDeathYear

    if (!indBirthYear) return 0.5

    // For parents, candidate must have been alive at child's birth
    if (relationshipType === 'father' || relationshipType === 'mother') {
      if (candDeathYear && candDeathYear < indBirthYear) {
        return 0.0  // Impossible - candidate died before individual was born
      }
      if (candBirthYear && candBirthYear > indBirthYear) {
        return 0.0  // Impossible - candidate born after individual
      }
    }

    // For spouse, should have overlapping lifespans
    if (relationshipType === 'spouse') {
      const indDeathYear = individual.estimatedDeathYear

      // Check for any overlap in lifetimes
      if (candDeathYear && indBirthYear && candDeathYear < indBirthYear + 18) {
        return 0.2  // Unlikely - candidate died before individual was marriageable
      }
      if (candBirthYear && indDeathYear && candBirthYear > indDeathYear) {
        return 0.0  // Impossible - no overlap
      }
    }

    return 1.0
  }

  /**
   * Calculate combined matching score
   */
  private calculateCombinedScore(
    factors: CandidateMatch['factors'],
    relationshipType: string
  ): number {
    // Weight factors differently based on relationship type
    let weights: Record<string, number>

    switch (relationshipType) {
      case 'father':
        weights = {
          surnameMatch: 0.35,
          ageCompatibility: 0.25,
          locationMatch: 0.15,
          networkProximity: 0.15,
          temporalPlausibility: 0.10
        }
        break

      case 'mother':
        weights = {
          surnameMatch: 0.20,  // Less important for mothers (maiden name)
          ageCompatibility: 0.30,
          locationMatch: 0.20,
          networkProximity: 0.20,
          temporalPlausibility: 0.10
        }
        break

      case 'spouse':
        weights = {
          surnameMatch: 0.10,
          ageCompatibility: 0.30,
          locationMatch: 0.25,
          networkProximity: 0.25,
          temporalPlausibility: 0.10
        }
        break

      case 'sibling':
        weights = {
          surnameMatch: 0.35,
          ageCompatibility: 0.25,
          locationMatch: 0.15,
          networkProximity: 0.20,
          temporalPlausibility: 0.05
        }
        break

      default:
        weights = {
          surnameMatch: 0.25,
          ageCompatibility: 0.25,
          locationMatch: 0.20,
          networkProximity: 0.20,
          temporalPlausibility: 0.10
        }
    }

    let score = 0
    for (const [factor, weight] of Object.entries(weights)) {
      score += (factors as any)[factor] * weight
    }

    return score
  }

  // ============================================================================
  // Helper Methods
  // ============================================================================

  private assessDataQuality(features: IndividualFeatures): number {
    // Count non-zero values in attribute mask
    const attributeMask = features.attributeMask
    const presentAttributes = attributeMask.filter(v => v > 0).length
    return presentAttributes / attributeMask.length
  }

  private isMarriageAgeRange(individual: Individual): boolean {
    const birthYear = individual.estimatedBirthYear
    if (!birthYear) return false

    const currentYear = new Date().getFullYear()
    const age = currentYear - birthYear

    // Typical marriage age range: 18-60
    return age >= 18 && age <= 80
  }

  private calculatePredictionConfidence(
    individual: Individual,
    features: IndividualFeatures
  ): number {
    let confidence = 0.5

    // Increase confidence if we have birth date
    if (individual.birth?.date) confidence += 0.15

    // Increase if we have place information
    if (individual.birth?.place) confidence += 0.1

    // Increase if we have name information
    if (individual.primaryName?.surname) confidence += 0.1

    // Increase based on graph connectivity
    const connectivity = (features.metadata.hasFather ? 1 : 0) +
      (features.metadata.hasMother ? 1 : 0) +
      features.metadata.numSiblings +
      (features.metadata.hasSpouse ? 1 : 0) +
      features.metadata.numChildren

    confidence += Math.min(connectivity / 10, 0.15)

    return Math.min(confidence, 1.0)
  }

  private calculateMatchConfidence(
    individual: Individual,
    candidate: Individual,
    factors: CandidateMatch['factors']
  ): number {
    // Base confidence from factor quality
    const factorValues = Object.values(factors)
    const avgFactor = factorValues.reduce((a, b) => a + b, 0) / factorValues.length

    // Adjust based on data availability
    let dataBonus = 0
    if (individual.estimatedBirthYear && candidate.estimatedBirthYear) dataBonus += 0.1
    if (individual.birth?.place && candidate.birth?.place) dataBonus += 0.1
    if (individual.primaryName?.surname && candidate.primaryName?.surname) dataBonus += 0.1

    return Math.min(avgFactor + dataBonus, 1.0)
  }

  private collectPlaces(individual: Individual): string[] {
    const places: string[] = []

    if (individual.birth?.place?.normalized) {
      places.push(individual.birth.place.normalized)
    }
    if (individual.death?.place?.normalized) {
      places.push(individual.death.place.normalized)
    }
    for (const res of individual.residences) {
      if (res.place?.normalized) {
        places.push(res.place.normalized)
      }
    }

    return places
  }

  private calculatePlaceMatch(place1: string, place2: string): number {
    const parts1 = place1.toLowerCase().split(',').map(s => s.trim())
    const parts2 = place2.toLowerCase().split(',').map(s => s.trim())

    if (place1.toLowerCase() === place2.toLowerCase()) return 1.0

    // Check component matches from most specific to least
    let matches = 0
    for (let i = 0; i < Math.min(parts1.length, parts2.length); i++) {
      if (parts1[i] === parts2[i]) matches++
    }

    // Weight earlier (more specific) matches higher
    if (matches === 0) return 0
    if (matches >= 3) return 0.9
    if (matches >= 2) return 0.7
    return 0.4
  }

  private calculateStringSimilarity(s1: string, s2: string): number {
    // Simple Levenshtein-based similarity
    const maxLen = Math.max(s1.length, s2.length)
    if (maxLen === 0) return 1.0

    const distance = this.levenshteinDistance(s1, s2)
    return 1 - distance / maxLen
  }

  private levenshteinDistance(s1: string, s2: string): number {
    const m = s1.length
    const n = s2.length
    const dp: number[][] = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0))

    for (let i = 0; i <= m; i++) dp[i][0] = i
    for (let j = 0; j <= n; j++) dp[0][j] = j

    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        if (s1[i - 1] === s2[j - 1]) {
          dp[i][j] = dp[i - 1][j - 1]
        } else {
          dp[i][j] = 1 + Math.min(
            dp[i - 1][j],     // deletion
            dp[i][j - 1],     // insertion
            dp[i - 1][j - 1]  // substitution
          )
        }
      }
    }

    return dp[m][n]
  }
}

// ============================================================================
// Convenience Functions
// ============================================================================

let defaultEngine: InferenceEngine | null = null

export function getInferenceEngine(config?: Partial<InferenceConfig>): InferenceEngine {
  if (!defaultEngine) {
    defaultEngine = new InferenceEngine(config)
  }
  return defaultEngine
}

export async function predictMissingRelatives(
  parsedGedcom: ParsedGedcom,
  config?: Partial<InferenceConfig>
): Promise<PredictionResult> {
  const engine = getInferenceEngine(config)
  return engine.predict(parsedGedcom)
}

/**
 * Format predictions for display/API response
 */
export function formatPredictionResults(result: PredictionResult): {
  summary: {
    totalIndividuals: number
    individualsWithMissingRelatives: number
    totalCandidatesFound: number
    processingTimeMs: number
  }
  topMissing: Array<{
    personId: string
    personName: string
    missingTypes: string[]
    confidence: number
  }>
  topCandidates: Array<{
    forPerson: string
    candidate: CandidateMatch
  }>
} {
  const individualsWithMissing = result.predictions.filter(
    p => p.overallMissingScore > 0.5
  )

  const totalCandidates = Array.from(result.candidates.values())
    .reduce((sum, arr) => sum + arr.length, 0)

  const topMissing = result.predictions
    .filter(p => p.overallMissingScore > 0.5)
    .sort((a, b) => b.overallMissingScore - a.overallMissingScore)
    .slice(0, 20)
    .map(p => {
      const missingTypes: string[] = []
      if (p.missingFather > 0.5) missingTypes.push('father')
      if (p.missingMother > 0.5) missingTypes.push('mother')
      if (p.missingSpouse > 0.5) missingTypes.push('spouse')

      return {
        personId: p.personId,
        personName: p.personName,
        missingTypes,
        confidence: p.confidence
      }
    })

  const topCandidates: Array<{ forPerson: string; candidate: CandidateMatch }> = []
  for (const [personId, candidates] of result.candidates) {
    for (const candidate of candidates.slice(0, 3)) {
      topCandidates.push({ forPerson: personId, candidate })
    }
  }
  topCandidates.sort((a, b) => b.candidate.score - a.candidate.score)

  return {
    summary: {
      totalIndividuals: result.predictions.length,
      individualsWithMissingRelatives: individualsWithMissing.length,
      totalCandidatesFound: totalCandidates,
      processingTimeMs: result.processingTimeMs
    },
    topMissing,
    topCandidates: topCandidates.slice(0, 30)
  }
}
