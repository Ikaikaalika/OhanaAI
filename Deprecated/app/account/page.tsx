import { getServerSession } from 'next-auth'
import { redirect } from 'next/navigation'
import { authOptions } from '@/lib/auth/config'

export default async function AccountPage() {
  const session = await getServerSession(authOptions)
  if (!session?.user?.id) redirect('/auth/signin')

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-2xl mx-auto p-6">
        <h1 className="text-2xl font-semibold mb-4">Account</h1>
        <div className="bg-white rounded-md border p-4 space-y-4">
          <div>
            <p className="text-sm text-gray-600">Signed in as</p>
            <p className="font-medium">{session.user?.email}</p>
          </div>
          <div className="border-t pt-4">
            <h2 className="text-lg font-semibold mb-2 text-red-700">Danger Zone</h2>
            <DeleteAccountButton />
          </div>
        </div>
      </div>
    </div>
  )
}

function DeleteAccountButton() {
  async function onDelete() {
    if (!confirm('Delete your account and all associated data? This cannot be undone.')) return
    const res = await fetch('/api/auth/user', { method: 'DELETE' })
    if (res.ok) {
      // Redirect to home or sign-in; session cookie will be invalid after deletion
      window.location.href = '/'
    } else {
      alert('Failed to delete account')
    }
  }
  return (
    <button onClick={onDelete} className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-md text-sm font-medium">
      Delete Account
    </button>
  )
}

