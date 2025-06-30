export function Features() {
  const features = [
    {
      name: 'AI-Powered Predictions',
      description: 'Our Graph Neural Network analyzes family relationships to predict missing parents with high accuracy.',
      icon: '🧠',
    },
    {
      name: 'GEDCOM File Support',
      description: 'Upload standard GEDCOM files from any genealogy software and get instant analysis.',
      icon: '📁',
    },
    {
      name: 'Interactive Family Trees',
      description: 'Visualize your complete family tree with predicted relationships highlighted.',
      icon: '🌳',
    },
    {
      name: 'Secure & Private',
      description: 'Your family data is encrypted and you have full control over its usage and deletion.',
      icon: '🔒',
    },
    {
      name: 'Continuous Learning',
      description: 'Our model improves over time as more data is processed, benefiting all users.',
      icon: '📈',
    },
    {
      name: 'Export Results',
      description: 'Download your enhanced family tree data in standard formats for use in other tools.',
      icon: '💾',
    },
  ]

  return (
    <div id="features" className="py-24 sm:py-32">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl lg:text-center">
          <h2 className="text-base font-semibold leading-7 text-indigo-600">Everything you need</h2>
          <p className="mt-2 text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
            Advanced Family Tree Intelligence
          </p>
          <p className="mt-6 text-lg leading-8 text-gray-600">
            Leverage cutting-edge machine learning to uncover hidden connections in your family history.
          </p>
        </div>
        <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-4xl">
          <dl className="grid max-w-xl grid-cols-1 gap-x-8 gap-y-10 lg:max-w-none lg:grid-cols-2 lg:gap-y-16">
            {features.map((feature) => (
              <div key={feature.name} className="relative pl-16">
                <dt className="text-base font-semibold leading-7 text-gray-900">
                  <div className="absolute left-0 top-0 flex h-10 w-10 items-center justify-center rounded-lg bg-indigo-600 text-white">
                    <span className="text-xl">{feature.icon}</span>
                  </div>
                  {feature.name}
                </dt>
                <dd className="mt-2 text-base leading-7 text-gray-600">{feature.description}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </div>
  )
}