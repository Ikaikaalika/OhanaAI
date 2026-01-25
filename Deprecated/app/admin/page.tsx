import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth/config'
import { db } from '@/lib/db'
import { mlTrainingStatus } from '@/lib/db/schema'
import { desc } from 'drizzle-orm'
import { existsSync, statSync, readdirSync } from 'fs'
import { join } from 'path'
import Link from 'next/link'

export const dynamic = 'force-dynamic'

export default async function AdminPage() {
  const session = await getServerSession(authOptions)
  if (!session?.user?.id) {
    return (
      <div className="p-6 max-w-3xl mx-auto">
        <h1 className="text-2xl font-semibold mb-4">Admin</h1>
        <p className="mb-4">You must be signed in to view this page.</p>
        <Link href="/auth/signin" className="text-blue-600 underline">Sign in</Link>
      </div>
    )
  }

  // Load latest training statuses
  const statuses = await db
    .select()
    .from(mlTrainingStatus)
    .orderBy(desc(mlTrainingStatus.updatedAt))
    .limit(20)

  // Model file presence and basic metadata
  const modelDir = join(process.cwd(), 'models', 'parent_predictor')
  const modelJson = join(modelDir, 'model.json')
  const modelPresent = existsSync(modelJson)
  let modelSizeKB = 0
  let files: { name: string; sizeKB: number }[] = []
  if (modelPresent) {
    try {
      const names = readdirSync(modelDir)
      files = names.map((name) => {
        const s = statSync(join(modelDir, name))
        return { name, sizeKB: Math.round(s.size / 1024) }
      })
      modelSizeKB = files.reduce((a, f) => a + f.sizeKB, 0)
    } catch {}
  }

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-8">
      <header className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Admin Dashboard</h1>
        <Link href="/" className="text-blue-600 underline">Back to app</Link>
      </header>

      <section className="border rounded-md p-4">
        <h2 className="text-xl font-semibold mb-2">Model Status</h2>
        <p className="mb-1">Present: <span className={modelPresent ? 'text-green-600' : 'text-red-600'}>{String(modelPresent)}</span></p>
        {modelPresent && (
          <>
            <p className="mb-1">Directory: <code>{modelDir}</code></p>
            <p className="mb-3">Total Size: {modelSizeKB} KB</p>
            <div className="overflow-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left border-b">
                    <th className="py-1 pr-4">File</th>
                    <th className="py-1">Size (KB)</th>
                  </tr>
                </thead>
                <tbody>
                  {files.map((f) => (
                    <tr key={f.name} className="border-b last:border-0">
                      <td className="py-1 pr-4"><code>{f.name}</code></td>
                      <td className="py-1">{f.sizeKB}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
        {!modelPresent && (
          <div className="mt-2 text-sm text-gray-600">
            No TFJS model found. Upload artifacts to <code>models/parent_predictor/</code> and redeploy.
          </div>
        )}
      </section>

      <section className="border rounded-md p-4">
        <h2 className="text-xl font-semibold mb-2">Latest Training Status</h2>
        {!statuses.length && <p>No status yet.</p>}
        {statuses.length > 0 && (
          <div className="overflow-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left border-b">
                  <th className="py-1 pr-4">Updated</th>
                  <th className="py-1 pr-4">Status</th>
                  <th className="py-1 pr-4">Model</th>
                  <th className="py-1">Message</th>
                </tr>
              </thead>
              <tbody>
                {statuses.map((row) => (
                  <tr key={row.id} className="border-b last:border-0 align-top">
                    <td className="py-1 pr-4">{new Date(row.updatedAt as unknown as string).toLocaleString()}</td>
                    <td className="py-1 pr-4"><span className="px-2 py-0.5 rounded bg-gray-100">{row.status}</span></td>
                    <td className="py-1 pr-4">{row.modelVersion || '-'}</td>
                    <td className="py-1 whitespace-pre-wrap">{row.message || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="text-sm text-gray-600">
        <p>To update this status programmatically, POST to <code>/api/ml/training-status</code> with your <code>ML_EXPORT_API_KEY</code>.</p>
      </section>
    </div>
  )
}

