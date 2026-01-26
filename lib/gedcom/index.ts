/**
 * OhanaAI - GEDCOM Parsing Module
 *
 * Exports all GEDCOM-related functionality for parsing and
 * processing genealogical data files.
 */

export {
  parseGedcom,
  getIndividualDisplayName,
  getAgeAtEvent,
  findCommonAncestors,
  type Individual,
  type Family,
  type Source,
  type Repository,
  type ParsedGedcom,
  type DateInfo,
  type PlaceInfo,
  type Name,
  type Event,
  type Occupation
} from './parser'
