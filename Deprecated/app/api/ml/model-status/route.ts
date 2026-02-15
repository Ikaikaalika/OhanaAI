import { NextResponse } from 'next/server'
import { getModelStatus } from '@/lib/ml/inference'

export const runtime = 'nodejs'

export async function GET() {
  const status = await getModelStatus()
  return NextResponse.json(status)
}
