import Image from 'next/image'

export function Features() {
  const features = [
    {
      name: 'AI-Powered Predictions',
      description: 'Our Graph Neural Network analyzes family relationships to predict missing parents with 94% accuracy using advanced machine learning.',
      icon: '🧠',
      gradient: 'from-blue-500 to-indigo-600',
    },
    {
      name: 'GEDCOM File Support',
      description: 'Upload standard GEDCOM files from any genealogy software (Ancestry, FamilySearch, MyHeritage) and get instant analysis.',
      icon: '📁',
      gradient: 'from-green-500 to-emerald-600',
    },
    {
      name: 'Interactive Family Trees',
      description: 'Visualize your complete family tree with predicted relationships highlighted. Zoom, explore, and discover connections.',
      icon: '🌳',
      gradient: 'from-amber-500 to-orange-600',
    },
    {
      name: 'Secure & Private',
      description: 'Your family data is encrypted end-to-end. You have full control over its usage and can delete it anytime.',
      icon: '🔒',
      gradient: 'from-red-500 to-pink-600',
    },
    {
      name: 'Continuous Learning',
      description: 'Our AI model improves over time as more families upload data, benefiting all users with better predictions.',
      icon: '📈',
      gradient: 'from-purple-500 to-violet-600',
    },
    {
      name: 'Export Results',
      description: 'Download your enhanced family tree data in standard GEDCOM format for use in other genealogy tools.',
      icon: '💾',
      gradient: 'from-cyan-500 to-blue-600',
    },
  ]

  return (
    <div id="features" className="py-24 sm:py-32 bg-white">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl lg:text-center">
          <h2 className="text-base font-semibold leading-7 text-indigo-600">Everything you need</h2>
          <p className="mt-2 text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
            Advanced Family Tree Intelligence
          </p>
          <p className="mt-6 text-lg leading-8 text-gray-600">
            Leverage cutting-edge machine learning to uncover hidden connections in your family history. 
            Our Graph Attention Network processes thousands of relationships to find missing links.
          </p>
        </div>

        {/* Features Grid */}
        <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none">
          <dl className="grid max-w-xl grid-cols-1 gap-x-8 gap-y-10 lg:max-w-none lg:grid-cols-2 xl:grid-cols-3 lg:gap-y-16">
            {features.map((feature, index) => (
              <div key={feature.name} className="relative group">
                <div className="relative bg-white rounded-2xl border border-gray-200 p-8 shadow-sm transition-all duration-300 hover:shadow-xl hover:-translate-y-1">
                  <dt>
                    <div className={`inline-flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-r ${feature.gradient} text-white shadow-lg`}>
                      <span className="text-2xl">{feature.icon}</span>
                    </div>
                    <h3 className="mt-6 text-lg font-semibold leading-7 text-gray-900 group-hover:text-indigo-600 transition-colors">
                      {feature.name}
                    </h3>
                  </dt>
                  <dd className="mt-4 text-base leading-7 text-gray-600">
                    {feature.description}
                  </dd>
                </div>
              </div>
            ))}
          </dl>
        </div>

        {/* How it Works Section */}
        <div className="mt-32">
          <div className="mx-auto max-w-2xl lg:text-center">
            <h2 className="text-base font-semibold leading-7 text-indigo-600">How it works</h2>
            <p className="mt-2 text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
              From Upload to Discovery in 3 Steps
            </p>
          </div>

          <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none">
            <div className="grid grid-cols-1 gap-y-16 lg:grid-cols-3 lg:gap-x-8">
              {[
                {
                  step: '01',
                  title: 'Upload Your GEDCOM',
                  description: 'Drag and drop your family tree file. We support all major genealogy formats.',
                  image: '/images/upload-step.jpg' // You can add this image
                },
                {
                  step: '02',
                  title: 'AI Analysis',
                  description: 'Our Graph Neural Network analyzes relationships and identifies missing connections.',
                  image: '/images/ai-analysis.jpg' // You can add this image
                },
                {
                  step: '03',
                  title: 'Discover Connections',
                  description: 'View predictions with confidence scores and explore your enhanced family tree.',
                  image: '/images/family-tree.jpg' // You can add this image
                }
              ].map((step, index) => (
                <div key={step.step} className="relative">
                  <div className="aspect-[16/9] w-full rounded-2xl bg-gradient-to-br from-gray-50 to-gray-100 shadow-lg ring-1 ring-gray-900/10">
                    {/* Placeholder for step images */}
                    <div className="flex items-center justify-center h-full">
                      <div className="text-center">
                        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-r from-indigo-500 to-purple-600 flex items-center justify-center">
                          <span className="text-2xl text-white font-bold">{step.step}</span>
                        </div>
                        <p className="text-gray-500">Step Image Placeholder</p>
                      </div>
                    </div>
                    {/* Uncomment when you add step images */}
                    {/* <Image
                      src={step.image}
                      alt={step.title}
                      fill
                      className="object-cover rounded-2xl"
                    /> */}
                  </div>
                  <div className="mt-6">
                    <h3 className="text-lg font-semibold text-gray-900">{step.title}</h3>
                    <p className="mt-2 text-gray-600">{step.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}