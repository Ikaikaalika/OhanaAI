import { NextResponse } from 'next/server'
import { join } from 'path'
import { existsSync } from 'fs'

export const runtime = 'nodejs'

export async function GET() {
  const modelPath = join(process.cwd(), 'models', 'parent_predictor', 'model.json')
  const present = existsSync(modelPath)
  return NextResponse.json({ present, path: present ? modelPath : null })
}

