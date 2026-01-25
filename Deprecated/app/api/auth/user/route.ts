import { NextRequest, NextResponse } from 'next/server'
import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth/config'
import { db } from '@/lib/db'
import { users, gedcomFiles } from '@/lib/db/schema'
import { eq } from 'drizzle-orm'
import { sendEmail, formatAdminEmail } from '@/lib/email/mailer'
import { deleteUploadFile } from '@/lib/storage'

export const runtime = 'nodejs'

export async function DELETE(_request: NextRequest) {
  try {
    const session = await getServerSession(authOptions)
    if (!session?.user?.id) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    // Collect user files to remove from disk
    const files = await db.select().from(gedcomFiles).where(eq(gedcomFiles.userId, session.user.id))

    // Delete files on disk best-effort
    for (const file of files) {
      try {
        await deleteUploadFile(file.filename)
      } catch {}
    }

    // Delete user (cascades will remove dependent rows)
    await db.delete(users).where(eq(users.id, session.user.id))

    // Notify admin
    const admin = process.env.ADMIN_EMAIL
    if (admin) {
      const payload = formatAdminEmail('User account deleted', [
        `User ID: ${session.user.id}`,
        `Time: ${new Date().toISOString()}`,
      ])
      sendEmail({ to: admin, ...payload }).catch(() => {})
    }

    return NextResponse.json({ message: 'Account deleted' }, { status: 200 })
  } catch (err) {
    return NextResponse.json({ error: 'Failed to delete account' }, { status: 500 })
  }
}
