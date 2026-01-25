'use client'

import { useState } from 'react'
import Link from 'next/link'
import { GedcomFile } from '@/lib/db/schema'

interface FileListProps {
  files: GedcomFile[]
}

export function FileList({ files }: FileListProps) {
  const [deletingId, setDeletingId] = useState<string | null>(null)

  const handleDelete = async (fileId: string) => {
    if (!confirm('Are you sure you want to delete this file? This action cannot be undone.')) {
      return
    }

    setDeletingId(fileId)
    
    try {
      const response = await fetch(`/api/gedcom/${fileId}`, {
        method: 'DELETE',
      })

      if (response.ok) {
        window.location.reload()
      } else {
        alert('Failed to delete file')
      }
    } catch (error) {
      alert('An error occurred while deleting the file')
    } finally {
      setDeletingId(null)
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(new Date(date))
  }

  return (
    <div className="bg-white shadow rounded-lg">
      <div className="px-6 py-4 border-b border-gray-200">
        <h3 className="text-lg font-medium text-gray-900">Your GEDCOM Files</h3>
        <p className="text-sm text-gray-500">
          {files.length} file{files.length !== 1 ? 's' : ''} uploaded
        </p>
      </div>

      {files.length === 0 ? (
        <div className="px-6 py-12 text-center">
          <div className="text-6xl mb-4">📂</div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No files uploaded yet</h3>
          <p className="text-gray-500 mb-4">
            Upload your first GEDCOM file to get started with AI-powered family tree analysis.
          </p>
        </div>
      ) : (
        <div className="divide-y divide-gray-200">
          {files.map((file) => (
            <div key={file.id} className="px-6 py-4 hover:bg-gray-50">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-3">
                    <div className="text-2xl">
                      {file.isProcessed ? '✅' : '⏳'}
                    </div>
                    <div>
                      <h4 className="text-sm font-medium text-gray-900">
                        {file.originalName}
                      </h4>
                      <div className="flex items-center space-x-4 text-sm text-gray-500">
                        <span>{formatFileSize(file.fileSize)}</span>
                        <span>Uploaded {formatDate(file.uploadedAt)}</span>
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          file.isProcessed
                            ? 'bg-green-100 text-green-800'
                            : 'bg-yellow-100 text-yellow-800'
                        }`}>
                          {file.isProcessed ? 'Processed' : 'Processing'}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-3">
                  {file.isProcessed && (
                    <Link
                      href={`/family-tree/${file.id}`}
                      className="text-indigo-600 hover:text-indigo-800 text-sm font-medium"
                    >
                      View Tree
                    </Link>
                  )}
                  <button
                    onClick={() => handleDelete(file.id)}
                    disabled={deletingId === file.id}
                    className="text-red-600 hover:text-red-800 text-sm font-medium disabled:opacity-50"
                  >
                    {deletingId === file.id ? 'Deleting...' : 'Delete'}
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}