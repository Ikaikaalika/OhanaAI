import { getServerSession } from 'next-auth'
import { redirect } from 'next/navigation'
import { authOptions } from '@/lib/auth/config'
import { db } from '@/lib/db'
import { gedcomFiles } from '@/lib/db/schema'
import { eq } from 'drizzle-orm'
import { FileUpload } from '@/components/dashboard/FileUpload'
import { FileList } from '@/components/dashboard/FileList'

export default async function Dashboard() {
  const session = await getServerSession(authOptions)

  if (!session?.user?.id) {
    redirect('/auth/signin')
  }

  const userFiles = await db
    .select()
    .from(gedcomFiles)
    .where(eq(gedcomFiles.userId, session.user.id))
    .orderBy(gedcomFiles.uploadedAt)

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="py-10">
        <header>
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <h1 className="text-3xl font-bold leading-tight text-gray-900">
              Dashboard
            </h1>
            <p className="mt-2 text-gray-600">
              Upload and manage your GEDCOM files
            </p>
          </div>
        </header>
        <main>
          <div className="max-w-7xl mx-auto sm:px-6 lg:px-8">
            <div className="px-4 py-8 sm:px-0">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2">
                  <FileList files={userFiles} />
                </div>
                <div>
                  <FileUpload />
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}