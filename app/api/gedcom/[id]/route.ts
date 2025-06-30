import { NextRequest, NextResponse } from 'next/server'
import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth/config'
import { db } from '@/lib/db'
import { gedcomFiles, familyTrees, mlTrainingData } from '@/lib/db/schema'
import { eq, and } from 'drizzle-orm'
import { unlink } from 'fs/promises'
import { join } from 'path'

export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const session = await getServerSession(authOptions)
    
    if (!session?.user?.id) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      )
    }

    const fileId = params.id

    // Get the file and verify ownership
    const file = await db
      .select()
      .from(gedcomFiles)
      .where(
        and(
          eq(gedcomFiles.id, fileId),
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

    // Delete related records first (due to foreign key constraints)
    await db.delete(mlTrainingData).where(eq(mlTrainingData.gedcomFileId, fileId))
    await db.delete(familyTrees).where(eq(familyTrees.gedcomFileId, fileId))
    
    // Delete the file record
    await db.delete(gedcomFiles).where(eq(gedcomFiles.id, fileId))

    // Delete the physical file
    try {
      const filepath = join(process.cwd(), 'uploads', file[0].filename)
      await unlink(filepath)
    } catch (error) {
      console.error('Error deleting physical file:', error)
      // Continue even if file deletion fails
    }

    return NextResponse.json({
      message: 'File deleted successfully'
    })

  } catch (error) {
    console.error('Delete error:', error)
    return NextResponse.json(
      { error: 'Failed to delete file' },
      { status: 500 }
    )
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const session = await getServerSession(authOptions)
    
    if (!session?.user?.id) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      )
    }

    const fileId = params.id

    // Get the file and verify ownership
    const file = await db
      .select()
      .from(gedcomFiles)
      .where(
        and(
          eq(gedcomFiles.id, fileId),
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

    // Get associated family tree
    const familyTree = await db
      .select()
      .from(familyTrees)
      .where(eq(familyTrees.gedcomFileId, fileId))
      .limit(1)

    return NextResponse.json({
      file: file[0],
      familyTree: familyTree[0] || null
    })

  } catch (error) {
    console.error('Get file error:', error)
    return NextResponse.json(
      { error: 'Failed to get file' },
      { status: 500 }
    )
  }
}