import Link from 'next/link'
import Image from 'next/image'

export function Hero() {
  return (
    <div className="relative isolate overflow-hidden bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Background decorative elements */}
      <div className="absolute inset-x-0 -top-40 -z-10 transform-gpu overflow-hidden blur-3xl sm:-top-80" aria-hidden="true">
        <div className="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-[#ff80b5] to-[#9089fc] opacity-20 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]" />
      </div>

      <div className="mx-auto max-w-7xl px-6 pb-24 pt-10 sm:pb-32 lg:flex lg:px-8 lg:py-40">
        <div className="mx-auto max-w-2xl flex-shrink-0 lg:mx-0 lg:max-w-xl lg:pt-8">
          <div className="mt-24 sm:mt-32 lg:mt-16">
            <a href="#" className="inline-flex space-x-6">
              <span className="rounded-full bg-indigo-500/10 px-3 py-1 text-sm font-semibold leading-6 text-indigo-600 ring-1 ring-inset ring-indigo-500/20">
                🧬 AI-Powered Genealogy
              </span>
            </a>
          </div>
          
          <h1 className="mt-10 text-4xl font-bold tracking-tight text-gray-900 sm:text-6xl">
            Discover Your 
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-purple-600"> Family History</span> with AI
          </h1>
          
          <p className="mt-6 text-lg leading-8 text-gray-600">
            Upload your GEDCOM files and let our advanced Graph Neural Network predict missing family connections. 
            Uncover hidden relationships and complete your family tree with unprecedented accuracy.
          </p>
          
          <div className="mt-10 flex items-center gap-x-6">
            <Link
              href="/auth/signup"
              className="rounded-md bg-indigo-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 transition-all duration-200 hover:scale-105"
            >
              Start Discovering
            </Link>
            <Link href="#features" className="text-sm font-semibold leading-6 text-gray-900 hover:text-indigo-600 transition-colors">
              Learn more <span aria-hidden="true">→</span>
            </Link>
          </div>

          {/* Stats */}
          <div className="mt-10 flex items-center gap-x-6 text-sm text-gray-600">
            <div className="flex items-center gap-x-2">
              <div className="flex -space-x-1">
                <div className="w-6 h-6 rounded-full bg-indigo-500 border-2 border-white"></div>
                <div className="w-6 h-6 rounded-full bg-purple-500 border-2 border-white"></div>
                <div className="w-6 h-6 rounded-full bg-pink-500 border-2 border-white"></div>
              </div>
              <span>8,620+ individuals analyzed</span>
            </div>
          </div>
        </div>

        <div className="mx-auto mt-16 flex max-w-2xl sm:mt-24 lg:ml-10 lg:mr-0 lg:mt-0 lg:max-w-none lg:flex-none xl:ml-32">
          <div className="max-w-3xl flex-none sm:max-w-5xl lg:max-w-none">
            {/* Hero Image - replace with your family tree image */}
            <div className="relative">
              <div className="aspect-[3/2] w-[76rem] rounded-2xl bg-gradient-to-br from-gray-50 to-gray-100 shadow-2xl ring-1 ring-gray-900/10">
                {/* Placeholder for your hero image */}
                <div className="flex items-center justify-center h-full">
                  <div className="text-center">
                    <div className="w-32 h-32 mx-auto mb-4 rounded-full bg-gradient-to-r from-indigo-500 to-purple-600 flex items-center justify-center">
                      <span className="text-4xl text-white">🌳</span>
                    </div>
                    <p className="text-gray-500 text-lg">Your Family Tree Image Here</p>
                    <p className="text-gray-400 text-sm mt-2">Place your image in public/images/hero-family-tree.jpg</p>
                  </div>
                </div>
                {/* Uncomment and use this when you add your image */}
                {/* <Image
                  src="/images/hero-family-tree.jpg"
                  alt="Family Tree Visualization"
                  fill
                  className="object-cover rounded-2xl"
                  priority
                /> */}
              </div>
              
              {/* Floating cards with stats */}
              <div className="absolute -bottom-6 left-6 bg-white rounded-lg shadow-lg p-4 border border-gray-100">
                <div className="text-2xl font-bold text-indigo-600">94%</div>
                <div className="text-sm text-gray-600">Prediction Accuracy</div>
              </div>
              
              <div className="absolute -top-6 right-6 bg-white rounded-lg shadow-lg p-4 border border-gray-100">
                <div className="text-2xl font-bold text-purple-600">2.5K+</div>
                <div className="text-sm text-gray-600">Families Connected</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom decorative element */}
      <div className="absolute inset-x-0 top-[calc(100%-13rem)] -z-10 transform-gpu overflow-hidden blur-3xl sm:top-[calc(100%-30rem)]" aria-hidden="true">
        <div className="relative left-[calc(50%+3rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 bg-gradient-to-tr from-[#ff80b5] to-[#9089fc] opacity-20 sm:left-[calc(50%+36rem)] sm:w-[72.1875rem]" />
      </div>
    </div>
  )
}