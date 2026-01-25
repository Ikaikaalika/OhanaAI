import { getServerSession } from 'next-auth'
import { redirect } from 'next/navigation'
import { authOptions } from '@/lib/auth/config'
import { db } from '@/lib/db'
import { gedcomFiles, familyTrees } from '@/lib/db/schema'
import { eq, and } from 'drizzle-orm'
import { FamilyTreeViewer } from '@/components/family-tree/FamilyTreeViewer'
import { notFound } from 'next/navigation'

interface PageProps {
  params: {
    id: string
  }
}

export default async function FamilyTreePage({ params }: PageProps) {
  const session = await getServerSession(authOptions)

  if (!session?.user?.id) {
    redirect('/auth/signin')
  }

  // Get the GEDCOM file and ensure it belongs to the current user
  const file = await db
    .select()
    .from(gedcomFiles)
    .where(
      and(
        eq(gedcomFiles.id, params.id),
        eq(gedcomFiles.userId, session.user.id)
      )
    )
    .limit(1)

  if (!file.length || !file[0].isProcessed) {
    notFound()
  }

  // Get the family tree data
  const familyTree = await db
    .select()
    .from(familyTrees)
    .where(eq(familyTrees.gedcomFileId, params.id))
    .limit(1)

  if (!familyTree.length) {
    notFound()
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="py-10">
        <header>
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold leading-tight text-gray-900">
                  Family Tree: {file[0].originalName}
                </h1>
                <p className="mt-2 text-gray-600">
                  Interactive visualization with AI-powered predictions
                </p>
              </div>
              <div className="flex items-center space-x-4">
                <button className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-md text-sm font-medium">
                  Export Data
                </button>
                <button className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-md text-sm font-medium">
                  Share Tree
                </button>
              </div>
            </div>
          </div>
        </header>
        <main>
          <div className="max-w-7xl mx-auto sm:px-6 lg:px-8">
            <div className="px-4 py-8 sm:px-0">
              <FamilyTreeViewer 
                familyTree={familyTree[0]}
                gedcomFile={file[0]}
              />
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}