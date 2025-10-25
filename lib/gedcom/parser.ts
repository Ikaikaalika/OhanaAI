export interface Individual {
  id: string
  name?: string
  firstName?: string
  lastName?: string
  gender?: 'M' | 'F'
  birthDate?: string
  birthPlace?: string
  deathDate?: string
  deathPlace?: string
  father?: string
  mother?: string
  spouse?: string[]
  children?: string[]
}

export interface Family {
  id: string
  husband?: string
  wife?: string
  children?: string[]
  marriageDate?: string
  marriagePlace?: string
}

export interface ParsedGedcom {
  individuals: Individual[]
  families: Family[]
  relationships: {
    parentChild: Array<{ parent: string; child: string }>
    spousal: Array<{ spouse1: string; spouse2: string }>
  }
}

export function parseGedcom(gedcomText: string): ParsedGedcom {
  const lines = gedcomText.split('\n').map(line => line.trim()).filter(line => line)
  
  const individuals: Individual[] = []
  const families: Family[] = []
  const relationships = {
    parentChild: [] as Array<{ parent: string; child: string }>,
    spousal: [] as Array<{ spouse1: string; spouse2: string }>
  }

  let currentRecord: any = null
  let currentType: 'INDI' | 'FAM' | null = null

  for (const line of lines) {
    const parts = line.split(' ')
    const level = parseInt(parts[0])
    const tag = parts[1]
    const value = parts.slice(2).join(' ')

    if (level === 0) {
      // Save previous record
      if (currentRecord && currentType) {
        if (currentType === 'INDI') {
          individuals.push(currentRecord)
        } else if (currentType === 'FAM') {
          families.push(currentRecord)
        }
      }

      const hasPointer = parts[1]?.startsWith('@')
      const pointer = hasPointer ? parts[1] : undefined
      const recordTag = hasPointer ? parts[2] : tag

      if (recordTag === 'INDI' && pointer) {
        currentRecord = { id: pointer }
        currentType = 'INDI'
      } else if (recordTag === 'FAM' && pointer) {
        currentRecord = { id: pointer }
        currentType = 'FAM'
      } else {
        currentRecord = null
        currentType = null
      }
    } else if (level === 1 && currentRecord) {
      switch (tag) {
        case 'NAME':
          if (currentType === 'INDI') {
            currentRecord.name = value.replace(/\//g, '')
            const nameParts = currentRecord.name.split(' ')
            currentRecord.firstName = nameParts[0]
            currentRecord.lastName = nameParts.slice(1).join(' ')
          }
          break
        case 'SEX':
          if (currentType === 'INDI') {
            currentRecord.gender = value === 'M' ? 'M' : 'F'
          }
          break
        case 'BIRT':
          if (currentType === 'INDI') {
            currentRecord._birthEvent = true
          }
          break
        case 'DEAT':
          if (currentType === 'INDI') {
            currentRecord._deathEvent = true
          }
          break
        case 'HUSB':
          if (currentType === 'FAM') {
            currentRecord.husband = value
          }
          break
        case 'WIFE':
          if (currentType === 'FAM') {
            currentRecord.wife = value
          }
          break
        case 'CHIL':
          if (currentType === 'FAM') {
            if (!currentRecord.children) currentRecord.children = []
            currentRecord.children.push(value)
          }
          break
        case 'MARR':
          if (currentType === 'FAM') {
            currentRecord._marriageEvent = true
          }
          break
      }
    } else if (level === 2 && currentRecord) {
      if (tag === 'DATE') {
        if (currentRecord._birthEvent) {
          currentRecord.birthDate = value
          currentRecord._birthEvent = false
        } else if (currentRecord._deathEvent) {
          currentRecord.deathDate = value
          currentRecord._deathEvent = false
        } else if (currentRecord._marriageEvent) {
          currentRecord.marriageDate = value
          currentRecord._marriageEvent = false
        }
      } else if (tag === 'PLAC') {
        if (currentRecord._birthEvent) {
          currentRecord.birthPlace = value
        } else if (currentRecord._deathEvent) {
          currentRecord.deathPlace = value
        } else if (currentRecord._marriageEvent) {
          currentRecord.marriagePlace = value
        }
      }
    }
  }

  // Save last record
  if (currentRecord && currentType) {
    if (currentType === 'INDI') {
      individuals.push(currentRecord)
    } else if (currentType === 'FAM') {
      families.push(currentRecord)
    }
  }

  // Build relationships
  for (const family of families) {
    // Parent-child relationships
    if (family.children) {
      for (const child of family.children) {
        if (family.husband) {
          relationships.parentChild.push({ parent: family.husband, child })
        }
        if (family.wife) {
          relationships.parentChild.push({ parent: family.wife, child })
        }
      }
    }

    // Spousal relationships
    if (family.husband && family.wife) {
      relationships.spousal.push({ spouse1: family.husband, spouse2: family.wife })
    }
  }

  // Add parent references to individuals
  for (const individual of individuals) {
    const parentRels = relationships.parentChild.filter(rel => rel.child === individual.id)
    const parents = parentRels.map(rel => rel.parent)
    
    for (const parent of parents) {
      const parentRecord = individuals.find(ind => ind.id === parent)
      if (parentRecord) {
        if (parentRecord.gender === 'M') {
          individual.father = parent
        } else if (parentRecord.gender === 'F') {
          individual.mother = parent
        }
      }
    }
  }

  return {
    individuals,
    families,
    relationships
  }
}
