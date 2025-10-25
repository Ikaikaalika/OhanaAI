import { NextRequest, NextResponse } from 'next/server'
import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth/config'
import { db } from '@/lib/db'
import { gedcomFiles, familyTrees } from '@/lib/db/schema'
import { eq, and } from 'drizzle-orm'
import { loadModel, runInference } from '@/lib/ml/inference'
import { sendEmail, formatAdminEmail } from '@/lib/email/mailer'

export const runtime = 'nodejs'

export async function POST(request: NextRequest) {
  try {
    const session = await getServerSession(authOptions)
    
    if (!session?.user?.id) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      )
    }

    const { gedcomFileId, personId } = await request.json()

    if (!gedcomFileId || !personId) {
      return NextResponse.json(
        { error: 'Missing required parameters' },
        { status: 400 }
      )
    }

    // Verify the file belongs to the user
    const file = await db
      .select()
      .from(gedcomFiles)
      .where(
        and(
          eq(gedcomFiles.id, gedcomFileId),
          eq(gedcomFiles.userId, session.user.id)
        )
      )
      .limit(1)

    if (!file.length) {
      return NextResponse.json(
        { error: 'File not found' },
        { status: 404 }
      )
    }

    // Get the family tree data
    const familyTree = await db
      .select()
      .from(familyTrees)
      .where(eq(familyTrees.gedcomFileId, gedcomFileId))
      .limit(1)

    if (!familyTree.length) {
      return NextResponse.json(
        { error: 'Family tree not found' },
        { status: 404 }
      )
    }

    // Check if model exists
    const modelExists = await loadModel()
    
    if (!modelExists) {
      return NextResponse.json({
        message: 'No trained model available yet. Please check back later as we continue to train our models.',
        predictions: [],
        modelStatus: 'training'
      })
    }

    // Run inference
    const individuals = familyTree[0].individuals as any[]
    const targetPerson = individuals.find((p: any) => p.id === personId)
    
    if (!targetPerson) {
      return NextResponse.json(
        { error: 'Person not found in family tree' },
        { status: 404 }
      )
    }

    const predictions = await runInference(familyTree[0], personId)

    const existingPredictions = normalizeStoredPredictions(file[0].predictions)
    const updatedPredictions = existingPredictions.filter(entry => entry.personId !== personId)
    if (Array.isArray(predictions) && predictions.length > 0) {
      updatedPredictions.push({
        personId,
        predictions
      })
    }

    await db.update(gedcomFiles)
      .set({ 
        predictions: updatedPredictions,
        modelVersion: 'v1.0' // Update this based on actual model version
      })
      .where(eq(gedcomFiles.id, gedcomFileId))

    // Notify when predictions occur for missing parents
    if (Array.isArray(predictions) && predictions.length > 0) {
      const admin = process.env.ADMIN_EMAIL
      if (admin) {
        const person = individuals.find((p: any) => p.id === personId)
        const payload = formatAdminEmail('New parent predictions generated', [
          `User ID: ${session.user.id}`,
          `GEDCOM File ID: ${gedcomFileId}`,
          `Target Person: ${person?.name || personId}`,
          `Predictions: ${predictions.map((p: any) => `${p.relationship}:${p.name}(${p.confidence})`).join(', ')}`,
          `Time: ${new Date().toISOString()}`,
        ])
        sendEmail({ to: admin, ...payload }).catch(() => {})
      }
    }

    return NextResponse.json({
      predictions,
      modelStatus: 'ready',
      message: 'Predictions generated successfully'
    })

  } catch (error) {
    console.error('Prediction error:', error)
    return NextResponse.json(
      { error: 'Failed to generate predictions' },
      { status: 500 }
    )
  }
}

type StoredPredictionEntry = {
  personId: string
  predictions: any[]
}

function normalizeStoredPredictions(payload: unknown): StoredPredictionEntry[] {
  if (!Array.isArray(payload)) return []
  return payload
    .filter((entry: any) => entry?.personId && Array.isArray(entry.predictions))
    .map((entry: any) => ({
      personId: entry.personId as string,
      predictions: entry.predictions
    }))
}
