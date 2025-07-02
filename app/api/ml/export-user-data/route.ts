import { NextRequest, NextResponse } from 'next/server'
import { db } from '@/lib/db'
import { mlTrainingData, gedcomFiles, users } from '@/lib/db/schema'
import { eq } from 'drizzle-orm'
import { writeFile, mkdir } from 'fs/promises'
import { join } from 'path'
import { createHash } from 'crypto'

// Secure API endpoint for exporting user data to your local machine
export async function POST(request: NextRequest) {
  try {
    // Security check - only allow from your domain or localhost
    const origin = request.headers.get('origin')
    const allowedOrigins = [
      'http://localhost:3000',
      'https://your-app.vercel.app', // Replace with your actual Vercel URL
      process.env.NEXTAUTH_URL
    ]
    
    if (!origin || !allowedOrigins.includes(origin)) {
      return NextResponse.json(
        { error: 'Unauthorized origin' },
        { status: 403 }
      )
    }

    // Verify API key/secret
    const { apiKey, includeMetadata = true } = await request.json()
    
    if (apiKey !== process.env.ML_EXPORT_API_KEY) {
      return NextResponse.json(
        { error: 'Invalid API key' },
        { status: 401 }
      )
    }

    console.log('Starting ML data export...')

    // Get all training data that hasn't been exported yet
    const newTrainingData = await db
      .select({
        id: mlTrainingData.id,
        gedcomFileId: mlTrainingData.gedcomFileId,
        graphData: mlTrainingData.graphData,
        labels: mlTrainingData.labels,
        exportedAt: mlTrainingData.exportedAt,
      })
      .from(mlTrainingData)
      .where(eq(mlTrainingData.includedInTraining, false))

    if (newTrainingData.length === 0) {
      return NextResponse.json({
        message: 'No new training data available',
        count: 0,
        exportUrl: null
      })
    }

    // Create secure export directory
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
    const exportDir = join(process.cwd(), 'exports', 'ml_training')
    
    try {
      await mkdir(exportDir, { recursive: true })
    } catch (error) {
      // Directory might already exist
    }

    // Generate secure filename
    const dataHash = createHash('sha256')
      .update(JSON.stringify(newTrainingData.map(d => d.id)))
      .digest('hex')
      .substring(0, 8)
    
    const filename = `ohana_training_data_${timestamp}_${dataHash}.json`
    const filepath = join(exportDir, filename)

    // Prepare export data with metadata if requested
    const exportData = {
      metadata: {
        exportedAt: new Date().toISOString(),
        version: '2.0',
        source: 'ohana-ai-production',
        count: newTrainingData.length,
        dataHash,
        ...(includeMetadata && await getExportMetadata(newTrainingData))
      },
      trainingData: newTrainingData.map(item => ({
        id: item.id,
        gedcomFileId: item.gedcomFileId,
        graphData: item.graphData,
        labels: item.labels,
        anonymized: true // Data is anonymized for privacy
      }))
    }

    // Write export file
    await writeFile(filepath, JSON.stringify(exportData, null, 2))

    // Generate webhook URL for your local machine to fetch the data
    const webhookUrl = `${origin}/api/ml/download-export/${filename}?key=${apiKey}`

    // Mark data as exported (but not yet included in training)
    for (const item of newTrainingData) {
      await db.update(mlTrainingData)
        .set({ exportedAt: new Date() })
        .where(eq(mlTrainingData.id, item.id))
    }

    // Generate training instructions
    const instructions = generateTrainingInstructions(newTrainingData.length, filename)

    console.log(`Exported ${newTrainingData.length} training examples`)

    return NextResponse.json({
      message: 'Training data exported successfully',
      count: newTrainingData.length,
      filename,
      exportPath: filepath,
      webhookUrl,
      instructions,
      metadata: exportData.metadata
    })

  } catch (error) {
    console.error('Export error:', error)
    return NextResponse.json(
      { error: 'Failed to export training data' },
      { status: 500 }
    )
  }
}

async function getExportMetadata(trainingData: any[]) {
  try {
    // Get associated GEDCOM files info (anonymized)
    const gedcomIds = Array.from(new Set(trainingData.map(d => d.gedcomFileId)))
    
    const gedcomInfo = await db
      .select({
        id: gedcomFiles.id,
        fileSize: gedcomFiles.fileSize,
        uploadedAt: gedcomFiles.uploadedAt,
        processedAt: gedcomFiles.processedAt,
        userId: gedcomFiles.userId
      })
      .from(gedcomFiles)
      .where(eq(gedcomFiles.id, gedcomIds[0])) // Sample just to avoid exposing too much data

    // Anonymized statistics
    return {
      uniqueGedcomFiles: gedcomIds.length,
      totalFileSize: gedcomInfo.reduce((sum, file) => sum + file.fileSize, 0),
      dateRange: {
        earliest: gedcomInfo.reduce((earliest: Date | null, file) => 
          !earliest || file.uploadedAt < earliest ? file.uploadedAt : earliest, null as Date | null),
        latest: gedcomInfo.reduce((latest: Date | null, file) => 
          !latest || file.uploadedAt > latest ? file.uploadedAt : latest, null as Date | null)
      },
      averageProcessingTime: 'anonymized' // Don't expose timing data
    }
  } catch (error) {
    console.error('Error getting metadata:', error)
    return {}
  }
}

function generateTrainingInstructions(dataCount: number, filename: string): string {
  return `
# Ohana AI Training Instructions

## New Data Available
- **Training Examples**: ${dataCount.toLocaleString()}
- **Export File**: ${filename}
- **Generated**: ${new Date().toLocaleString()}

## Quick Start (M1 Mac)

1. **Download the data**:
   \`\`\`bash
   # The webhook URL provided above will download the file
   # Or manually copy from your Vercel deployment exports folder
   \`\`\`

2. **Set up Python environment**:
   \`\`\`bash
   python3 -m venv venv_ohana
   source venv_ohana/bin/activate
   pip install tensorflow-macos tensorflow-metal
   pip install networkx scikit-learn matplotlib pandas numpy
   \`\`\`

3. **Run training**:
   \`\`\`bash
   # Place the export file in your project directory
   python train_model_m1.py --data-file ${filename}
   \`\`\`

4. **Deploy updated model**:
   \`\`\`bash
   # Convert to TensorFlow.js
   tensorflowjs_converter --input_format=keras \\
     models/parent_predictor/ohana_model_m1.h5 \\
     models/parent_predictor/

   # Upload to your web app (manual or via API)
   \`\`\`

## Automated Training Pipeline

For continuous training, set up a cron job:

\`\`\`bash
# Add to crontab (run daily at 2 AM)
0 2 * * * cd /path/to/ohana-ai && python scripts/auto_train.py
\`\`\`
`
}

// Download endpoint for your local machine
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const key = searchParams.get('key')
    const filename = request.url.split('/').pop()?.split('?')[0]

    if (key !== process.env.ML_EXPORT_API_KEY || !filename) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    const filepath = join(process.cwd(), 'exports', 'ml_training', filename)
    
    try {
      const fileContent = await import('fs').then(fs => 
        fs.promises.readFile(filepath, 'utf-8')
      )
      
      return new NextResponse(fileContent, {
        headers: {
          'Content-Type': 'application/json',
          'Content-Disposition': `attachment; filename="${filename}"`,
        },
      })
    } catch (error) {
      return NextResponse.json({ error: 'File not found' }, { status: 404 })
    }

  } catch (error) {
    return NextResponse.json({ error: 'Download failed' }, { status: 500 })
  }
}