import { NextRequest, NextResponse } from 'next/server'
import { db } from '@/lib/db'
import { mlTrainingData } from '@/lib/db/schema'
import { inArray } from 'drizzle-orm'
import { del } from '@vercel/blob'

export const runtime = 'nodejs'

export async function POST(request: NextRequest) {
  try {
    const { apiKey, exportedIds, exportBlobPath, exportUrl } = await request.json()

    if (!apiKey || apiKey !== process.env.ML_EXPORT_API_KEY) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    if (!Array.isArray(exportedIds) || exportedIds.length === 0) {
      return NextResponse.json({ ok: false, error: 'No exportedIds provided' }, { status: 400 })
    }

    await db
      .update(mlTrainingData)
      .set({ includedInTraining: true })
      .where(inArray(mlTrainingData.id, exportedIds))

    // Best-effort cleanup for the exported blob
    if (exportBlobPath || exportUrl) {
      try {
        await del(exportUrl || exportBlobPath)
      } catch {}
    }

    return NextResponse.json({ ok: true, updated: exportedIds.length })
  } catch (error) {
    return NextResponse.json({ error: 'Invalid request' }, { status: 400 })
  }
}
