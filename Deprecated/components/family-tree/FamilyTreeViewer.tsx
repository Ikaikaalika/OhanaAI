'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { FamilyTree, GedcomFile } from '@/lib/db/schema'

interface FamilyTreeViewerProps {
  familyTree: FamilyTree
  gedcomFile: GedcomFile
}

type Person = Record<string, any>

type AncestorNode = {
  id: string
  name: string
  person: Person
  relationLabel: string
  children: AncestorNode[]
}

type Prediction = {
  personId?: string
  relationship: string
  name: string
  confidence?: number
}

type PredictionGroup = {
  personId: string
  predictions: Prediction[]
}

export function FamilyTreeViewer({ familyTree, gedcomFile }: FamilyTreeViewerProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [selectedPerson, setSelectedPerson] = useState<Person | null>(null)
  const [rootPersonId, setRootPersonId] = useState<string | null>(null)

  const individuals = useMemo(() => {
    return (familyTree.individuals as Person[]) || []
  }, [familyTree.individuals])
  const personMap = useMemo(() => {
    const map = new Map<string, Person>()
    individuals.forEach(person => {
      if (person?.id) map.set(person.id, person)
    })
    return map
  }, [individuals])

  useEffect(() => {
    if (!rootPersonId && individuals.length > 0) {
      setRootPersonId(individuals[0].id)
      setSelectedPerson(individuals[0])
    }
  }, [rootPersonId, individuals])

  const predictionsMap = useMemo(() => {
    const entries = Array.isArray(gedcomFile.predictions) ? (gedcomFile.predictions as PredictionGroup[]) : []
    const map = new Map<string, Prediction[]>()
    for (const group of entries) {
      if (!group?.personId || !Array.isArray(group.predictions)) continue
      map.set(group.personId, group.predictions)
    }
    return map
  }, [gedcomFile.predictions])

  const ancestorTree = useMemo(() => {
    if (!rootPersonId) return null
    const root = personMap.get(rootPersonId)
    if (!root) return null
    return buildAncestorTree(root, personMap, predictionsMap)
  }, [rootPersonId, personMap, predictionsMap])

  useEffect(() => {
    if (!ancestorTree || !svgRef.current) return

    const width = 840
    const height = 840
    const radius = (Math.min(width, height) / 2) - 40

    const root = d3.hierarchy(ancestorTree)
    const tree = d3.tree<AncestorNode>().size([2 * Math.PI, radius])
    tree(root)

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()
    svg.attr('viewBox', `${-width / 2} ${-height / 2} ${width} ${height}`)

    const linkGenerator = d3.linkRadial()
      .angle((d: any) => d.x)
      .radius((d: any) => d.y)

    svg.append('g')
      .selectAll('path')
      .data(root.links())
      .join('path')
      .attr('d', linkGenerator as any)
      .attr('stroke', '#CBD5F5')
      .attr('fill', 'none')
      .attr('stroke-width', 1.5)

    const nodeGroup = svg.append('g')
      .selectAll('g')
      .data(root.descendants())
      .join('g')
      .attr('transform', (d) => `rotate(${((d.x ?? 0) * 180) / Math.PI - 90}) translate(${d.y ?? 0},0)`)

    nodeGroup.append('circle')
      .attr('r', (d) => d.depth === 0 ? 9 : 5)
      .attr('fill', (d) => getNodeColor(d.data))
      .attr('stroke', '#1D4ED8')
      .attr('stroke-width', 1.5)
      .style('cursor', 'pointer')
      .on('click', (_, d) => {
        setSelectedPerson(d.data.person)
        if (d.depth === 0) return
        setRootPersonId(d.data.id.startsWith('virtual-') ? rootPersonId : d.data.id)
      })

    nodeGroup.append('text')
      .attr('dy', '0.31em')
      .attr('x', (d) => (d.x ?? 0) < Math.PI ? 10 : -10)
      .attr('text-anchor', (d) => (d.x ?? 0) < Math.PI ? 'start' : 'end')
      .attr('transform', (d) => (d.x ?? 0) >= Math.PI ? 'rotate(180)' : '')
      .text((d) => formatLabel(d.data))
      .style('font-size', '12px')
      .style('fill', '#1F2937')
      .style('cursor', 'pointer')
      .on('click', (_, d) => {
        setSelectedPerson(d.data.person)
        if (d.depth === 0) return
        setRootPersonId(d.data.id.startsWith('virtual-') ? rootPersonId : d.data.id)
      })

  }, [ancestorTree, rootPersonId])

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden">
      <div className="flex flex-col lg:flex-row h-[720px]">
        <div className="flex-1 flex flex-col">
          <div className="px-6 py-4 border-b border-gray-100 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h3 className="text-lg font-semibold">Ancestor Fan Chart</h3>
              <p className="text-sm text-gray-500">Select any person to re-center the chart</p>
            </div>
            <div>
              <select
                className="border rounded-md px-3 py-2 text-sm"
                value={rootPersonId || ''}
                onChange={(e) => {
                  const person = personMap.get(e.target.value)
                  setRootPersonId(e.target.value)
                  if (person) setSelectedPerson(person)
                }}
              >
                {individuals.map(person => (
                  <option key={person.id} value={person.id}>
                    {person.name || 'Unknown'}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="flex-1 flex items-center justify-center">
            <svg ref={svgRef} className="w-full h-full" />
          </div>
        </div>
        <div className="w-full lg:w-96 bg-gray-50 border-t lg:border-t-0 lg:border-l border-gray-200 p-6 overflow-y-auto">
          <h3 className="text-lg font-semibold mb-4">Details</h3>
          {selectedPerson ? (
            <div className="space-y-4">
              <div>
                <h4 className="text-xl font-semibold text-gray-900">{selectedPerson.name || 'Unknown Person'}</h4>
                <p className="text-sm text-gray-500 mt-1">ID: {selectedPerson.id}</p>
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                {selectedPerson.birthDate && (
                  <div>
                    <p className="text-gray-500">Born</p>
                    <p className="font-medium text-gray-900">{selectedPerson.birthDate}</p>
                  </div>
                )}
                {selectedPerson.deathDate && (
                  <div>
                    <p className="text-gray-500">Died</p>
                    <p className="font-medium text-gray-900">{selectedPerson.deathDate}</p>
                  </div>
                )}
                {selectedPerson.birthPlace && (
                  <div>
                    <p className="text-gray-500">Birth Place</p>
                    <p className="font-medium text-gray-900">{selectedPerson.birthPlace}</p>
                  </div>
                )}
                {selectedPerson.deathPlace && (
                  <div>
                    <p className="text-gray-500">Death Place</p>
                    <p className="font-medium text-gray-900">{selectedPerson.deathPlace}</p>
                  </div>
                )}
              </div>
              <div>
                <h5 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">Parents</h5>
                <div className="mt-2 space-y-1 text-sm">
                  <p className={selectedPerson.father ? 'text-gray-900' : 'text-red-600'}>
                    Father: {renderParentName(selectedPerson.father, personMap)}
                  </p>
                  <p className={selectedPerson.mother ? 'text-gray-900' : 'text-red-600'}>
                    Mother: {renderParentName(selectedPerson.mother, personMap)}
                  </p>
                </div>
              </div>
              <div>
                <h5 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">Predicted Relatives</h5>
                <PredictionsList
                  entries={predictionsMap.get(selectedPerson.id || '') || []}
                />
              </div>
            </div>
          ) : (
            <p className="text-sm text-gray-500">Select a person in the fan chart to view their details.</p>
          )}
        </div>
      </div>
    </div>
  )
}

function buildAncestorTree(
  person: Person,
  personMap: Map<string, Person>,
  predictions: Map<string, Prediction[]>,
  relationLabel = 'Self',
  visited = new Set<string>()
): AncestorNode {
  if (person.id) visited.add(person.id)

  const parentIds = [person.father, person.mother].filter(Boolean) as string[]
  const children: AncestorNode[] = []

  for (const parentId of parentIds) {
    if (visited.has(parentId)) continue
    const parent = personMap.get(parentId)
    if (parent) {
      children.push(
        buildAncestorTree(
          parent,
          personMap,
          predictions,
          parent.gender === 'F' ? 'Mother' : 'Father',
          new Set(visited)
        )
      )
    }
  }

  if (!parentIds.length) {
    const predictedParents = predictions.get(person.id) || []
    predictedParents.forEach(entry => {
      children.push({
        id: `virtual-${person.id}-${entry.relationship}-${entry.name}`,
        name: `${entry.name} (AI)`,
        relationLabel: entry.relationship,
        person: {
          id: `virtual-${person.id}-${entry.relationship}`,
          name: entry.name,
          confidence: entry.confidence,
        },
        children: []
      })
    })
  }

  return {
    id: person.id,
    name: person.name || 'Unknown',
    person,
    relationLabel,
    children
  }
}

function getNodeColor(node: AncestorNode) {
  if (node.id.startsWith('virtual-')) return '#FDE68A'
  if (node.relationLabel === 'Mother') return '#FCE7F3'
  if (node.relationLabel === 'Father') return '#DBEAFE'
  if (node.relationLabel === 'Self') return '#C7D2FE'
  return '#E0E7FF'
}

function formatLabel(node: AncestorNode) {
  const base = node.name || 'Unknown'
  if (node.relationLabel === 'Self') return base
  return `${node.relationLabel}: ${base}`
}

function renderParentName(parentId: string | undefined, personMap: Map<string, Person>) {
  if (!parentId) return 'Unknown'
  const parent = personMap.get(parentId)
  return parent?.name || 'Unknown'
}

function PredictionsList({ entries }: { entries: Prediction[] }) {
  if (!entries.length) {
    return <p className="text-sm text-gray-500">No stored predictions yet for this person.</p>
  }

  return (
    <div className="space-y-2">
      {entries.map((entry, idx) => (
        <div key={`${entry.personId}-${entry.relationship}-${idx}`} className="bg-white rounded border border-indigo-100 p-3 text-sm">
          <p className="font-medium text-gray-900">
            {entry.relationship}: {entry.name}
          </p>
          {entry.confidence !== undefined && (
            <p className="text-indigo-700">
              Confidence: {(entry.confidence * 100).toFixed(1)}%
            </p>
          )}
        </div>
      ))}
    </div>
  )
}
