import { NextRequest, NextResponse } from 'next/server'
import { db } from '@/lib/db'
import { mlTrainingStatus } from '@/lib/db/schema'
import { desc } from 'drizzle-orm'
import { enqueueGlobalModelRefresh } from '@/lib/jobs/modelRefresh'

export const runtime = 'nodejs'

export async function POST(request: NextRequest) {
  try {
    const { apiKey, status, message, modelVersion, details } = await request.json()

    if (!apiKey || apiKey !== process.env.ML_EXPORT_API_KEY) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    const normalizedStatus = String(status || '')
    const normalizedModelVersion = modelVersion ? String(modelVersion) : null

    await db.insert(mlTrainingStatus).values({
      status: normalizedStatus,
      message: message ? String(message) : null,
      modelVersion: normalizedModelVersion,
      details: details ? details : null,
    })

    const statusLower = normalizedStatus.toLowerCase()
    if (statusLower === 'ready' || statusLower === 'completed') {
      enqueueGlobalModelRefresh({
        trigger: `training-status:${statusLower}`,
        modelVersion: normalizedModelVersion
      })
    }

    return NextResponse.json({ ok: true })
  } catch (error) {
    return NextResponse.json({ error: 'Invalid request' }, { status: 400 })
  }
}

export async function GET() {
  try {
    const rows = await db
      .select()
      .from(mlTrainingStatus)
      .orderBy(desc(mlTrainingStatus.updatedAt))
      .limit(1)

    if (!rows.length) {
      return NextResponse.json({ status: 'unknown' })
    }
    return NextResponse.json(rows[0])
  } catch (error) {
    return NextResponse.json({ status: 'unknown' })
  }
}
