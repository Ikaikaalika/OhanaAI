/**
 * Enhanced GEDCOM Parser
 * Extracts comprehensive genealogical data for ML training
 */

// ============================================================================
// Core Data Types
// ============================================================================

export interface DateInfo {
  raw: string
  year?: number
  month?: number
  day?: number
  circa?: boolean  // Approximate date (ABT, EST, CAL)
  range?: { start?: string; end?: string }  // BET...AND, FROM...TO
}

export interface PlaceInfo {
  raw: string
  components: string[]  // Split by comma: [city, county, state, country]
  normalized?: string
}

export interface Event {
  type: string
  date?: DateInfo
  place?: PlaceInfo
  description?: string
  age?: string
  cause?: string  // For death events
  witnesses?: string[]
}

export interface Name {
  full: string
  given?: string      // First/given name
  surname?: string    // Family name
  prefix?: string     // Mr., Dr., etc.
  suffix?: string     // Jr., Sr., III, etc.
  nickname?: string
  maiden?: string     // Maiden name
  married?: string    // Married name
  type?: 'birth' | 'married' | 'aka' | 'immigrant' | 'religious'
}

export interface Occupation {
  title: string
  date?: DateInfo
  place?: PlaceInfo
}

export interface Individual {
  id: string
  names: Name[]
  primaryName?: Name
  gender?: 'M' | 'F' | 'U'

  // Life events
  birth?: Event
  death?: Event
  baptism?: Event
  burial?: Event
  christening?: Event

  // Marriage/family events
  marriages: Event[]
  divorces: Event[]

  // Other events
  residences: Event[]
  occupations: Occupation[]
  education: Event[]
  immigration: Event[]
  emigration: Event[]
  naturalization: Event[]
  military: Event[]
  census: Event[]

  // Attributes
  religion?: string
  ethnicity?: string
  nationality?: string
  socialSecurityNumber?: string
  healthConditions: string[]
  physicalDescription?: string

  // Relationships (IDs)
  father?: string
  mother?: string
  spouses: string[]
  children: string[]
  siblings: string[]  // Computed

  // Source references
  sources: string[]
  notes: string[]

  // Computed fields
  estimatedBirthYear?: number
  estimatedDeathYear?: number
  lifespan?: number
}

export interface Family {
  id: string
  husband?: string
  wife?: string
  children: string[]

  marriage?: Event
  divorce?: Event
  annulment?: Event
  engagement?: Event

  notes: string[]
  sources: string[]
}

export interface Source {
  id: string
  title?: string
  author?: string
  publisher?: string
  date?: string
  repository?: string
  citation?: string
}

export interface Repository {
  id: string
  name?: string
  address?: string
}

export interface ParsedGedcom {
  individuals: Map<string, Individual>
  families: Map<string, Family>
  sources: Map<string, Source>
  repositories: Map<string, Repository>

  // Relationship indices for fast lookup
  relationships: {
    parentChild: Array<{ parent: string; child: string; type: 'father' | 'mother' }>
    spousal: Array<{ spouse1: string; spouse2: string; familyId: string }>
    sibling: Array<{ sibling1: string; sibling2: string }>
  }

  // Statistics
  stats: {
    totalIndividuals: number
    totalFamilies: number
    dateRange: { earliest?: number; latest?: number }
    generations: number
    locations: Set<string>
  }
}

// ============================================================================
// Parser Implementation
// ============================================================================

interface GedcomLine {
  level: number
  tag: string
  pointer?: string
  value?: string
}

function parseLine(line: string): GedcomLine | null {
  const trimmed = line.trim()
  if (!trimmed) return null

  // GEDCOM line format: LEVEL [POINTER] TAG [VALUE]
  const match = trimmed.match(/^(\d+)\s+(?:(@[^@]+@)\s+)?(\S+)(?:\s+(.*))?$/)
  if (!match) return null

  return {
    level: parseInt(match[1], 10),
    pointer: match[2],
    tag: match[3],
    value: match[4]?.trim()
  }
}

function parseDate(raw: string): DateInfo {
  const result: DateInfo = { raw }

  // Handle approximate dates
  if (raw.match(/^(ABT|ABOUT|EST|CAL|CIRCA)\s+/i)) {
    result.circa = true
    raw = raw.replace(/^(ABT|ABOUT|EST|CAL|CIRCA)\s+/i, '')
  }

  // Handle date ranges: BET date AND date
  const betMatch = raw.match(/^BET\s+(.+)\s+AND\s+(.+)$/i)
  if (betMatch) {
    result.range = { start: betMatch[1], end: betMatch[2] }
    raw = betMatch[1] // Use start date for year extraction
  }

  // Handle FROM...TO
  const fromToMatch = raw.match(/^FROM\s+(.+)\s+TO\s+(.+)$/i)
  if (fromToMatch) {
    result.range = { start: fromToMatch[1], end: fromToMatch[2] }
    raw = fromToMatch[1]
  }

  // Handle BEF/AFT dates
  raw = raw.replace(/^(BEF|AFT|BEFORE|AFTER)\s+/i, '')

  // Extract year
  const yearMatch = raw.match(/\b(\d{4})\b/)
  if (yearMatch) {
    result.year = parseInt(yearMatch[1], 10)
  }

  // Extract month
  const months: Record<string, number> = {
    JAN: 1, FEB: 2, MAR: 3, APR: 4, MAY: 5, JUN: 6,
    JUL: 7, AUG: 8, SEP: 9, OCT: 10, NOV: 11, DEC: 12
  }
  const monthMatch = raw.match(/\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b/i)
  if (monthMatch) {
    result.month = months[monthMatch[1].toUpperCase()]
  }

  // Extract day
  const dayMatch = raw.match(/\b(\d{1,2})\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)/i)
  if (dayMatch) {
    result.day = parseInt(dayMatch[1], 10)
  }

  return result
}

function parsePlace(raw: string): PlaceInfo {
  const components = raw.split(',').map(s => s.trim()).filter(Boolean)
  return {
    raw,
    components,
    normalized: components.join(', ')
  }
}

function parseName(value: string): Name {
  const result: Name = { full: value.replace(/\//g, '').trim() }

  // Extract surname (between slashes in GEDCOM)
  const surnameMatch = value.match(/\/([^/]+)\//)
  if (surnameMatch) {
    result.surname = surnameMatch[1].trim()
  }

  // Extract given name (before the first slash)
  const givenMatch = value.match(/^([^/]+)\//)
  if (givenMatch) {
    result.given = givenMatch[1].trim()
  }

  // Extract suffix (after the closing slash)
  const suffixMatch = value.match(/\/\s*(\S.*)$/)
  if (suffixMatch && !suffixMatch[1].startsWith('/')) {
    const suffix = suffixMatch[1].trim()
    if (suffix && !suffix.match(/^\//)) {
      result.suffix = suffix
    }
  }

  return result
}

export function parseGedcom(gedcomText: string): ParsedGedcom {
  const lines = gedcomText.split(/\r?\n/)

  const individuals = new Map<string, Individual>()
  const families = new Map<string, Family>()
  const sources = new Map<string, Source>()
  const repositories = new Map<string, Repository>()

  // Parser state
  type RecordType = 'INDI' | 'FAM' | 'SOUR' | 'REPO' | null
  let currentRecord: any = null
  let currentType: RecordType = null
  let currentEvent: Event | null = null
  let eventStack: string[] = []  // Track nested event context

  function createEmptyIndividual(id: string): Individual {
    return {
      id,
      names: [],
      marriages: [],
      divorces: [],
      residences: [],
      occupations: [],
      education: [],
      immigration: [],
      emigration: [],
      naturalization: [],
      military: [],
      census: [],
      healthConditions: [],
      spouses: [],
      children: [],
      siblings: [],
      sources: [],
      notes: []
    }
  }

  function createEmptyFamily(id: string): Family {
    return {
      id,
      children: [],
      notes: [],
      sources: []
    }
  }

  function saveCurrentRecord() {
    if (!currentRecord || !currentType) return

    // Finalize current event if any
    if (currentEvent && currentType === 'INDI') {
      const eventType = eventStack[eventStack.length - 1]
      if (eventType) {
        switch (eventType) {
          case 'BIRT': currentRecord.birth = currentEvent; break
          case 'DEAT': currentRecord.death = currentEvent; break
          case 'BAPM': case 'CHR': currentRecord.baptism = currentEvent; break
          case 'BURI': currentRecord.burial = currentEvent; break
          case 'RESI': currentRecord.residences.push(currentEvent); break
          case 'IMMI': currentRecord.immigration.push(currentEvent); break
          case 'EMIG': currentRecord.emigration.push(currentEvent); break
          case 'NATU': currentRecord.naturalization.push(currentEvent); break
          case 'CENS': currentRecord.census.push(currentEvent); break
          case 'EDUC': currentRecord.education.push(currentEvent); break
          case 'MILI': currentRecord.military.push(currentEvent); break
        }
      }
    }

    switch (currentType) {
      case 'INDI':
        // Set primary name
        if (currentRecord.names.length > 0) {
          currentRecord.primaryName = currentRecord.names[0]
        }
        individuals.set(currentRecord.id, currentRecord)
        break
      case 'FAM':
        families.set(currentRecord.id, currentRecord)
        break
      case 'SOUR':
        sources.set(currentRecord.id, currentRecord)
        break
      case 'REPO':
        repositories.set(currentRecord.id, currentRecord)
        break
    }

    currentRecord = null
    currentType = null
    currentEvent = null
    eventStack = []
  }

  for (const rawLine of lines) {
    const parsed = parseLine(rawLine)
    if (!parsed) continue

    const { level, tag, pointer, value } = parsed

    // Level 0: New record
    if (level === 0) {
      saveCurrentRecord()

      if (pointer) {
        const recordTag = tag
        if (value === 'INDI' || recordTag === 'INDI') {
          const id = pointer || value
          currentRecord = createEmptyIndividual(id)
          currentType = 'INDI'
        } else if (value === 'FAM' || recordTag === 'FAM') {
          const id = pointer || value
          currentRecord = createEmptyFamily(id)
          currentType = 'FAM'
        } else if (value === 'SOUR' || recordTag === 'SOUR') {
          currentRecord = { id: pointer }
          currentType = 'SOUR'
        } else if (value === 'REPO' || recordTag === 'REPO') {
          currentRecord = { id: pointer }
          currentType = 'REPO'
        }
      }
      continue
    }

    if (!currentRecord || !currentType) continue

    // Level 1: Main attributes and events
    if (level === 1) {
      // Finalize previous event if switching
      if (currentEvent && currentType === 'INDI') {
        const eventType = eventStack[eventStack.length - 1]
        if (eventType) {
          switch (eventType) {
            case 'BIRT': currentRecord.birth = currentEvent; break
            case 'DEAT': currentRecord.death = currentEvent; break
            case 'BAPM': case 'CHR': currentRecord.baptism = currentEvent; break
            case 'BURI': currentRecord.burial = currentEvent; break
            case 'RESI': currentRecord.residences.push(currentEvent); break
            case 'IMMI': currentRecord.immigration.push(currentEvent); break
            case 'EMIG': currentRecord.emigration.push(currentEvent); break
            case 'NATU': currentRecord.naturalization.push(currentEvent); break
            case 'CENS': currentRecord.census.push(currentEvent); break
            case 'EDUC': currentRecord.education.push(currentEvent); break
            case 'MILI': currentRecord.military.push(currentEvent); break
          }
        }
        currentEvent = null
        eventStack = []
      }

      if (currentType === 'INDI') {
        switch (tag) {
          case 'NAME':
            const name = parseName(value || '')
            currentRecord.names.push(name)
            break
          case 'SEX':
            currentRecord.gender = value === 'M' ? 'M' : value === 'F' ? 'F' : 'U'
            break
          case 'BIRT':
          case 'DEAT':
          case 'BAPM':
          case 'CHR':
          case 'BURI':
          case 'RESI':
          case 'IMMI':
          case 'EMIG':
          case 'NATU':
          case 'CENS':
          case 'EDUC':
          case 'MILI':
            currentEvent = { type: tag }
            eventStack.push(tag)
            break
          case 'OCCU':
            currentRecord.occupations.push({ title: value || '' })
            break
          case 'RELI':
            currentRecord.religion = value
            break
          case 'NOTE':
            if (value) currentRecord.notes.push(value)
            break
          case 'SOUR':
            if (value) currentRecord.sources.push(value)
            break
          case 'FAMS':
            // Family where this person is a spouse
            if (value) currentRecord.spouses.push(value)  // Will resolve later
            break
          case 'FAMC':
            // Family where this person is a child
            // Will be used to set parents
            break
        }
      } else if (currentType === 'FAM') {
        switch (tag) {
          case 'HUSB':
            currentRecord.husband = value
            break
          case 'WIFE':
            currentRecord.wife = value
            break
          case 'CHIL':
            if (value) currentRecord.children.push(value)
            break
          case 'MARR':
            currentEvent = { type: 'MARR' }
            eventStack.push('MARR')
            break
          case 'DIV':
            currentEvent = { type: 'DIV' }
            eventStack.push('DIV')
            break
          case 'ANUL':
            currentEvent = { type: 'ANUL' }
            eventStack.push('ANUL')
            break
          case 'NOTE':
            if (value) currentRecord.notes.push(value)
            break
          case 'SOUR':
            if (value) currentRecord.sources.push(value)
            break
        }
      } else if (currentType === 'SOUR') {
        switch (tag) {
          case 'TITL':
            currentRecord.title = value
            break
          case 'AUTH':
            currentRecord.author = value
            break
          case 'PUBL':
            currentRecord.publisher = value
            break
        }
      } else if (currentType === 'REPO') {
        switch (tag) {
          case 'NAME':
            currentRecord.name = value
            break
          case 'ADDR':
            currentRecord.address = value
            break
        }
      }
    }

    // Level 2: Event details
    if (level === 2 && currentEvent) {
      switch (tag) {
        case 'DATE':
          currentEvent.date = parseDate(value || '')
          break
        case 'PLAC':
          currentEvent.place = parsePlace(value || '')
          break
        case 'AGE':
          currentEvent.age = value
          break
        case 'CAUS':
          currentEvent.cause = value
          break
        case 'NOTE':
          currentEvent.description = value
          break
      }
    }

    // Handle family events saving
    if (level === 1 && currentType === 'FAM' && currentEvent) {
      const eventType = eventStack[eventStack.length - 1]
      if (eventType === 'MARR') {
        currentRecord.marriage = currentEvent
      } else if (eventType === 'DIV') {
        currentRecord.divorce = currentEvent
      } else if (eventType === 'ANUL') {
        currentRecord.annulment = currentEvent
      }
    }
  }

  // Save last record
  saveCurrentRecord()

  // Build relationships
  const relationships: ParsedGedcom['relationships'] = {
    parentChild: [],
    spousal: [],
    sibling: []
  }

  // Process families to build relationships
  for (const [familyId, family] of families) {
    // Parent-child relationships
    for (const childId of family.children) {
      if (family.husband) {
        relationships.parentChild.push({
          parent: family.husband,
          child: childId,
          type: 'father'
        })
        const child = individuals.get(childId)
        if (child) child.father = family.husband
      }
      if (family.wife) {
        relationships.parentChild.push({
          parent: family.wife,
          child: childId,
          type: 'mother'
        })
        const child = individuals.get(childId)
        if (child) child.mother = family.wife
      }
    }

    // Spousal relationships
    if (family.husband && family.wife) {
      relationships.spousal.push({
        spouse1: family.husband,
        spouse2: family.wife,
        familyId
      })

      // Update individual spouse lists with actual person IDs
      const husband = individuals.get(family.husband)
      const wife = individuals.get(family.wife)
      if (husband) {
        const idx = husband.spouses.indexOf(familyId)
        if (idx >= 0) husband.spouses[idx] = family.wife
        else if (!husband.spouses.includes(family.wife)) husband.spouses.push(family.wife)
      }
      if (wife) {
        const idx = wife.spouses.indexOf(familyId)
        if (idx >= 0) wife.spouses[idx] = family.husband
        else if (!wife.spouses.includes(family.husband)) wife.spouses.push(family.husband)
      }
    }

    // Sibling relationships
    for (let i = 0; i < family.children.length; i++) {
      for (let j = i + 1; j < family.children.length; j++) {
        relationships.sibling.push({
          sibling1: family.children[i],
          sibling2: family.children[j]
        })

        // Update individuals
        const sib1 = individuals.get(family.children[i])
        const sib2 = individuals.get(family.children[j])
        if (sib1 && !sib1.siblings.includes(family.children[j])) {
          sib1.siblings.push(family.children[j])
        }
        if (sib2 && !sib2.siblings.includes(family.children[i])) {
          sib2.siblings.push(family.children[i])
        }
      }
    }

    // Update children lists on individuals
    for (const childId of family.children) {
      if (family.husband) {
        const parent = individuals.get(family.husband)
        if (parent && !parent.children.includes(childId)) {
          parent.children.push(childId)
        }
      }
      if (family.wife) {
        const parent = individuals.get(family.wife)
        if (parent && !parent.children.includes(childId)) {
          parent.children.push(childId)
        }
      }
    }
  }

  // Compute estimated birth/death years and lifespans
  const allYears: number[] = []
  for (const [, individual] of individuals) {
    if (individual.birth?.date?.year) {
      individual.estimatedBirthYear = individual.birth.date.year
      allYears.push(individual.birth.date.year)
    }
    if (individual.death?.date?.year) {
      individual.estimatedDeathYear = individual.death.date.year
      allYears.push(individual.death.date.year)
    }
    if (individual.estimatedBirthYear && individual.estimatedDeathYear) {
      individual.lifespan = individual.estimatedDeathYear - individual.estimatedBirthYear
    }
  }

  // Collect all locations
  const locations = new Set<string>()
  for (const [, individual] of individuals) {
    if (individual.birth?.place?.normalized) locations.add(individual.birth.place.normalized)
    if (individual.death?.place?.normalized) locations.add(individual.death.place.normalized)
    for (const res of individual.residences) {
      if (res.place?.normalized) locations.add(res.place.normalized)
    }
  }

  // Estimate generations (simple heuristic)
  let generations = 1
  const visited = new Set<string>()
  function countGenerations(id: string, depth: number) {
    if (visited.has(id) || depth > 50) return
    visited.add(id)
    generations = Math.max(generations, depth)
    const ind = individuals.get(id)
    if (!ind) return
    for (const childId of ind.children) {
      countGenerations(childId, depth + 1)
    }
  }
  // Find roots (people with no parents)
  for (const [id, ind] of individuals) {
    if (!ind.father && !ind.mother) {
      countGenerations(id, 1)
    }
  }

  return {
    individuals,
    families,
    sources,
    repositories,
    relationships,
    stats: {
      totalIndividuals: individuals.size,
      totalFamilies: families.size,
      dateRange: {
        earliest: allYears.length > 0 ? Math.min(...allYears) : undefined,
        latest: allYears.length > 0 ? Math.max(...allYears) : undefined
      },
      generations,
      locations
    }
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

export function getIndividualDisplayName(individual: Individual): string {
  if (individual.primaryName) {
    return individual.primaryName.full
  }
  if (individual.names.length > 0) {
    return individual.names[0].full
  }
  return individual.id
}

export function getAgeAtEvent(individual: Individual, eventYear?: number): number | undefined {
  if (!individual.estimatedBirthYear || !eventYear) return undefined
  return eventYear - individual.estimatedBirthYear
}

export function findCommonAncestors(
  ind1: Individual,
  ind2: Individual,
  individuals: Map<string, Individual>
): string[] {
  const ancestors1 = new Set<string>()
  const ancestors2 = new Set<string>()

  function collectAncestors(id: string | undefined, set: Set<string>, depth: number) {
    if (!id || depth > 20 || set.has(id)) return
    set.add(id)
    const ind = individuals.get(id)
    if (!ind) return
    collectAncestors(ind.father, set, depth + 1)
    collectAncestors(ind.mother, set, depth + 1)
  }

  collectAncestors(ind1.father, ancestors1, 0)
  collectAncestors(ind1.mother, ancestors1, 0)
  collectAncestors(ind2.father, ancestors2, 0)
  collectAncestors(ind2.mother, ancestors2, 0)

  const common: string[] = []
  for (const id of ancestors1) {
    if (ancestors2.has(id)) common.push(id)
  }

  return common
}
