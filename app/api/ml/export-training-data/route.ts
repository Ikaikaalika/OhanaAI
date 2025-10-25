import { NextRequest, NextResponse } from 'next/server'
import { writeFile } from 'fs/promises'
import { join } from 'path'
import { exportPendingTrainingData, generateRequirements, generateTrainingScript } from '@/lib/ml/exporter'

export const runtime = 'nodejs'

// This endpoint should be secured or called via a cron job
export async function POST(request: NextRequest) {
  try {
    // Basic security check - in production, use proper authentication
    const { authorization } = await request.json()
    
    if (authorization !== process.env.EXPORT_SECRET) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      )
    }

    const exportResult = await exportPendingTrainingData()

    if (exportResult.count === 0) {
      return NextResponse.json({
        message: 'No new training data to export',
        count: 0
      })
    }

    const trainingDir = exportResult.directory
    const baseDir = process.cwd()

    const trainingScript = generateTrainingScript(exportResult.count)
    await writeFile(join(trainingDir, 'run_training.py'), trainingScript)
    
    const requirements = generateRequirements()
    await writeFile(join(trainingDir, 'requirements.txt'), requirements)

    // Optional: trigger local training automatically
    if (process.env.AUTO_TRAIN === 'true') {
      try {
        const { spawn } = await import('child_process')
        const command = process.env.TRAINING_COMMAND || 'python3'
        const args = process.env.TRAINING_ARGS
          ? process.env.TRAINING_ARGS.split(' ')
          : ['train_model_mlx.py', '--data-dir', trainingDir]
        const py = spawn(command, args, { cwd: baseDir, stdio: 'ignore', detached: true })
        py.unref()
      } catch (err) {
        console.error('Auto-train spawn error:', err)
      }
    }

    return NextResponse.json({
      message: 'Training data exported successfully',
      count: exportResult.count,
      batches: exportResult.batches,
      directory: trainingDir,
      autoTrain: process.env.AUTO_TRAIN === 'true'
    })

  } catch (error) {
    console.error('Export error:', error)
    return NextResponse.json(
      { error: 'Failed to export training data' },
      { status: 500 }
    )
  }

}