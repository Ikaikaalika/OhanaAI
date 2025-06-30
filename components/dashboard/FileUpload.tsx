'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { useRouter } from 'next/navigation'

export function FileUpload() {
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const router = useRouter()

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (!file) return

    setUploading(true)
    setError('')

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('/api/gedcom/upload', {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        setError(data.error || 'Upload failed')
      } else {
        router.refresh()
      }
    } catch (error) {
      setError('An error occurred during upload')
    } finally {
      setUploading(false)
    }
  }, [router])

  const {
    getRootProps,
    getInputProps,
    isDragActive,
    fileRejections,
  } = useDropzone({
    onDrop,
    accept: {
      'application/octet-stream': ['.ged'],
      'text/plain': ['.ged', '.gedcom'],
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
  })

  return (
    <div className="bg-white shadow rounded-lg p-6">
      <h3 className="text-lg font-medium text-gray-900 mb-4">
        Upload GEDCOM File
      </h3>
      
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors
          ${isDragActive
            ? 'border-indigo-500 bg-indigo-50'
            : 'border-gray-300 hover:border-indigo-400'
          }
          ${uploading ? 'pointer-events-none opacity-50' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        <div className="space-y-4">
          <div className="text-6xl">📁</div>
          
          {uploading ? (
            <div>
              <p className="text-lg font-medium text-gray-900">Uploading...</p>
              <p className="text-sm text-gray-500">Please wait while we process your file</p>
            </div>
          ) : isDragActive ? (
            <div>
              <p className="text-lg font-medium text-indigo-600">Drop your GEDCOM file here</p>
            </div>
          ) : (
            <div>
              <p className="text-lg font-medium text-gray-900">
                Drag & drop your GEDCOM file here
              </p>
              <p className="text-sm text-gray-500">
                or click to select a file (max 10MB)
              </p>
              <p className="text-xs text-gray-400 mt-2">
                Supported formats: .ged, .gedcom
              </p>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="mt-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}

      {fileRejections.length > 0 && (
        <div className="mt-4 bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded">
          <ul className="list-disc list-inside text-sm">
            {fileRejections.map(({ file, errors }, index) => (
              <li key={index}>
                {file.name}: {errors.map(e => e.message).join(', ')}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}