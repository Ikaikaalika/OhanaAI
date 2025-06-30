import { Navbar } from '@/components/ui/Navbar'
import { Hero } from '@/components/ui/Hero'
import { Features } from '@/components/ui/Features'

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <Navbar />
      <Hero />
      <Features />
    </main>
  )
}