'use client'

import { useEffect, useRef, useState } from 'react'
import { Network } from 'vis-network'
import { DataSet } from 'vis-data'
import { FamilyTree, GedcomFile } from '@/lib/db/schema'

interface FamilyTreeViewerProps {
  familyTree: FamilyTree
  gedcomFile: GedcomFile
}

export function FamilyTreeViewer({ familyTree, gedcomFile }: FamilyTreeViewerProps) {
  const networkRef = useRef<HTMLDivElement>(null)
  const [network, setNetwork] = useState<Network | null>(null)
  const [selectedPerson, setSelectedPerson] = useState<any>(null)
  const [predictions, setPredictions] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!networkRef.current || !familyTree.individuals) return

    const individuals = familyTree.individuals as any[]
    const relationships = familyTree.relationships as any

    // Create nodes for visualization
    const nodes = new DataSet(
      individuals.map(person => ({
        id: person.id,
        label: person.name || 'Unknown',
        title: createPersonTooltip(person),
        color: {
          background: getPersonColor(person),
          border: '#2B5CE6',
          highlight: {
            background: '#FFE4B5',
            border: '#FF8C00'
          }
        },
        shape: 'box',
        font: { size: 12, face: 'Arial' },
        widthConstraint: { minimum: 80, maximum: 150 }
      }))
    )

    // Create edges for relationships
    const edges = new DataSet([
      ...relationships.parentChild?.map((rel: any) => ({
        from: rel.parent,
        to: rel.child,
        arrows: 'to',
        color: { color: '#848484' },
        width: 2,
        label: 'parent'
      })) || [],
      ...relationships.spousal?.map((rel: any) => ({
        from: rel.spouse1,
        to: rel.spouse2,
        color: { color: '#FF6B6B' },
        width: 3,
        label: 'spouse',
        dashes: false
      })) || []
    ])

    const options = {
      layout: {
        hierarchical: {
          enabled: true,
          direction: 'UD',
          sortMethod: 'directed',
          levelSeparation: 150,
          nodeSpacing: 200,
          treeSpacing: 200
        }
      },
      physics: {
        enabled: false
      },
      nodes: {
        borderWidth: 2,
        shadow: true
      },
      edges: {
        smooth: {
          enabled: true,
          type: 'dynamic',
          roundness: 0.5
        }
      },
      interaction: {
        dragNodes: true,
        dragView: true,
        zoomView: true
      }
    }

    const net = new Network(networkRef.current, { nodes, edges }, options)
    
    net.on('click', (params) => {
      if (params.nodes.length > 0) {
        const personId = params.nodes[0]
        const person = individuals.find(p => p.id === personId)
        setSelectedPerson(person)
      }
    })

    setNetwork(net)

    return () => {
      net.destroy()
    }
  }, [familyTree])

  const createPersonTooltip = (person: any) => {
    let tooltip = `<b>${person.name || 'Unknown'}</b><br/>`
    if (person.birthDate) tooltip += `Born: ${person.birthDate}<br/>`
    if (person.deathDate) tooltip += `Died: ${person.deathDate}<br/>`
    if (person.gender) tooltip += `Gender: ${person.gender === 'M' ? 'Male' : 'Female'}<br/>`
    return tooltip
  }

  const getPersonColor = (person: any) => {
    if (!person.father && !person.mother) return '#FFE4E1' // Missing both parents
    if (!person.father || !person.mother) return '#FFF8DC' // Missing one parent
    return '#E6F3FF' // Has both parents
  }

  const runPredictions = async () => {
    if (!selectedPerson) return
    
    setLoading(true)
    try {
      const response = await fetch('/api/ml/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          gedcomFileId: gedcomFile.id,
          personId: selectedPerson.id
        })
      })

      if (response.ok) {
        const data = await response.json()
        setPredictions(data.predictions)
      } else {
        console.error('Prediction failed')
      }
    } catch (error) {
      console.error('Error running predictions:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden">
      <div className="flex h-[600px]">
        {/* Main tree visualization */}
        <div className="flex-1 relative">
          <div ref={networkRef} className="w-full h-full" />
          
          {/* Legend */}
          <div className="absolute top-4 left-4 bg-white p-4 rounded-lg shadow-md">
            <h4 className="font-semibold mb-2">Legend</h4>
            <div className="space-y-2 text-sm">
              <div className="flex items-center">
                <div className="w-4 h-4 bg-red-100 border border-red-300 rounded mr-2"></div>
                <span>Missing both parents</span>
              </div>
              <div className="flex items-center">
                <div className="w-4 h-4 bg-yellow-100 border border-yellow-300 rounded mr-2"></div>
                <span>Missing one parent</span>
              </div>
              <div className="flex items-center">
                <div className="w-4 h-4 bg-blue-100 border border-blue-300 rounded mr-2"></div>
                <span>Has both parents</span>
              </div>
            </div>
          </div>
        </div>

        {/* Side panel */}
        <div className="w-80 bg-gray-50 border-l border-gray-200 p-6">
          <h3 className="text-lg font-semibold mb-4">Person Details</h3>
          
          {selectedPerson ? (
            <div className="space-y-4">
              <div>
                <h4 className="font-medium text-gray-900">
                  {selectedPerson.name || 'Unknown Name'}
                </h4>
                <p className="text-sm text-gray-500">ID: {selectedPerson.id}</p>
              </div>

              <div className="space-y-2">
                {selectedPerson.birthDate && (
                  <p className="text-sm"><strong>Born:</strong> {selectedPerson.birthDate}</p>
                )}
                {selectedPerson.deathDate && (
                  <p className="text-sm"><strong>Died:</strong> {selectedPerson.deathDate}</p>
                )}
                {selectedPerson.gender && (
                  <p className="text-sm"><strong>Gender:</strong> {selectedPerson.gender === 'M' ? 'Male' : 'Female'}</p>
                )}
              </div>

              <div>
                <h5 className="font-medium text-gray-900 mb-2">Parents</h5>
                <div className="space-y-1 text-sm">
                  {selectedPerson.father ? (
                    <p>Father: {selectedPerson.father}</p>
                  ) : (
                    <p className="text-red-600">Father: Unknown</p>
                  )}
                  {selectedPerson.mother ? (
                    <p>Mother: {selectedPerson.mother}</p>
                  ) : (
                    <p className="text-red-600">Mother: Unknown</p>
                  )}
                </div>
              </div>

              {(!selectedPerson.father || !selectedPerson.mother) && (
                <div>
                  <button
                    onClick={runPredictions}
                    disabled={loading}
                    className="w-full bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-md text-sm font-medium disabled:opacity-50"
                  >
                    {loading ? 'Predicting...' : 'Predict Missing Parents'}
                  </button>
                </div>
              )}

              {predictions && (
                <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                  <h5 className="font-medium text-blue-900 mb-2">AI Predictions</h5>
                  {predictions.length > 0 ? (
                    <div className="space-y-2">
                      {predictions.map((prediction: any, index: number) => (
                        <div key={index} className="text-sm">
                          <p className="font-medium">
                            Predicted {prediction.relationship}: {prediction.name}
                          </p>
                          <p className="text-blue-700">
                            Confidence: {(prediction.confidence * 100).toFixed(1)}%
                          </p>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-blue-700 text-sm">No strong predictions found</p>
                  )}
                </div>
              )}
            </div>
          ) : (
            <p className="text-gray-500 text-sm">
              Click on a person in the family tree to view details and run AI predictions.
            </p>
          )}
        </div>
      </div>
    </div>
  )
}