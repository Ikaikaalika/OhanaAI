import { NextRequest, NextResponse } from 'next/server'
import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth/config'
import { db } from '@/lib/db'
import { gedcomFiles } from '@/lib/db/schema'
import { parseGedcom } from '@/lib/gedcom/parser'
import { v4 as uuidv4 } from 'uuid'
import { saveUploadFile } from '@/lib/storage'
import { enqueueGedcomProcessing, processGedcomImmediately } from '@/lib/jobs/gedcomProcessor'
import { createHash } from 'crypto'
import { and, eq } from 'drizzle-orm'

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

    const fileHash = createHash('sha256').update(buffer).digest('hex')

    const duplicate = await db
      .select({ id: gedcomFiles.id })
      .from(gedcomFiles)
      .where(
        and(
          eq(gedcomFiles.userId, session.user.id),
          eq(gedcomFiles.fileHash, fileHash)
        )
      )
      .limit(1)

    if (duplicate.length) {
      return NextResponse.json(
        {
          error: 'Duplicate upload detected. This GEDCOM file already exists in your library.',
          fileId: duplicate[0].id
        },
        { status: 409 }
      )
    }

    // Generate unique filename
    const filename = `${uuidv4()}.ged`
    const blobPath = `uploads/${session.user.id}/${filename}`
    const stored = await saveUploadFile(filename, buffer, {
      blobPath,
      contentType: 'text/plain'
    })

    // Parse GEDCOM file
    const gedcomText = buffer.toString('utf-8')
    const parsedData = parseGedcom(gedcomText)

    // Create database record
    const [newFile] = await db.insert(gedcomFiles).values({
      userId: session.user.id,
      filename,
      originalName: file.name,
      fileSize: file.size,
      fileHash,
      blobUrl: stored.blobUrl || null,
      blobPath: stored.blobPath || null,
      parsedData,
      isProcessed: false,
    }).returning()

    const runSynchronously = (process.env.SYNC_GEDCOM_PROCESSING ?? 'true') !== 'false'

    if (runSynchronously) {
      await processGedcomImmediately({ fileId: newFile.id, parsedData })
      return NextResponse.json(
        {
          message: 'File uploaded and processed successfully. Predictions are ready.',
          fileId: newFile.id
        },
        { status: 201 }
      )
    }

    enqueueGedcomProcessing({
      fileId: newFile.id,
      parsedData,
    })

    return NextResponse.json(
      {
        message: 'File uploaded successfully. Processing will continue in the background.',
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
