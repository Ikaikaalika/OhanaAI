import { NextRequest, NextResponse } from 'next/server'
import { put } from '@vercel/blob'

export const runtime = 'nodejs'

function sanitizePath(input: string) {
  return input.replace(/^\/*/, '').replace(/\.\./g, '').replace(/\\/g, '/')
}

export async function POST(request: NextRequest) {
  try {
    const form = await request.formData()
    const apiKey = form.get('apiKey')?.toString()
    const modelVersion = form.get('modelVersion')?.toString()
    const path = form.get('path')?.toString()
    const file = form.get('file') as File | null

    if (!apiKey || apiKey !== process.env.ML_EXPORT_API_KEY) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    if (!modelVersion || !path || !file) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 })
    }

    const safePath = sanitizePath(path)
    const blobPath = `tfjs/${modelVersion}/${safePath}`
    const bytes = Buffer.from(await file.arrayBuffer())
    const blob = await put(blobPath, bytes, {
      access: 'public',
      contentType: file.type || 'application/octet-stream',
    })

    return NextResponse.json({
      url: blob.url,
      path: blob.pathname || blobPath,
    })
  } catch (error) {
    return NextResponse.json({ error: 'Upload failed' }, { status: 500 })
  }
}
