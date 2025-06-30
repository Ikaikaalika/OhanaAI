import { NextRequest, NextResponse } from 'next/server'
import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth/config'
import { db } from '@/lib/db'
import { gedcomFiles, familyTrees, mlTrainingData } from '@/lib/db/schema'
import { eq } from 'drizzle-orm'
import { parseGedcom } from '@/lib/gedcom/parser'
import { processGedcomForML } from '@/lib/ml/dataProcessor'
import { v4 as uuidv4 } from 'uuid'
import { writeFile, mkdir } from 'fs/promises'
import { join } from 'path'

export async function POST(request: NextRequest) {
  try {
    const session = await getServerSession(authOptions)
    
    if (!session?.user?.id) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      )
    }

    const formData = await request.formData()
    const file = formData.get('file') as File
    
    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      )
    }

    if (!file.name.match(/\.(ged|gedcom)$/i)) {
      return NextResponse.json(
        { error: 'Invalid file type. Please upload a .ged or .gedcom file' },
        { status: 400 }
      )
    }

    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      return NextResponse.json(
        { error: 'File too large. Maximum size is 10MB' },
        { status: 400 }
      )
    }

    const bytes = await file.arrayBuffer()
    const buffer = Buffer.from(bytes)
    
    // Create uploads directory if it doesn't exist
    const uploadsDir = join(process.cwd(), 'uploads')
    try {
      await mkdir(uploadsDir, { recursive: true })
    } catch (error) {
      // Directory might already exist
    }

    // Generate unique filename
    const filename = `${uuidv4()}.ged`
    const filepath = join(uploadsDir, filename)
    
    // Save file to disk
    await writeFile(filepath, buffer)

    // Parse GEDCOM file
    const gedcomText = buffer.toString('utf-8')
    const parsedData = parseGedcom(gedcomText)

    // Create database record
    const [newFile] = await db.insert(gedcomFiles).values({
      userId: session.user.id,
      filename,
      originalName: file.name,
      fileSize: file.size,
      parsedData,
      isProcessed: false,
    }).returning()

    // Process asynchronously
    processFileAsync(newFile.id, parsedData)

    return NextResponse.json(
      { 
        message: 'File uploaded successfully',
        fileId: newFile.id 
      },
      { status: 201 }
    )

  } catch (error) {
    console.error('Upload error:', error)
    return NextResponse.json(
      { error: 'Failed to upload file' },
      { status: 500 }
    )
  }
}

async function processFileAsync(fileId: string, parsedData: any) {
  try {
    // Create family tree structure
    const individuals = parsedData.individuals || []
    const relationships = parsedData.relationships || []

    // Create family tree record
    await db.insert(familyTrees).values({
      gedcomFileId: fileId,
      individuals,
      relationships,
    })

    // Process for ML training
    const mlData = processGedcomForML(parsedData)
    
    if (mlData) {
      await db.insert(mlTrainingData).values({
        gedcomFileId: fileId,
        graphData: mlData.graphData,
        labels: mlData.labels,
      })
    }

    // Mark as processed
    await db.update(gedcomFiles)
      .set({ 
        isProcessed: true,
        processedAt: new Date(),
      })
      .where(eq(gedcomFiles.id, fileId))

  } catch (error) {
    console.error('Processing error:', error)
  }
}