/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    outputFileTracingIncludes: {
      // Ensure model files are bundled with the predict API route
      'app/api/ml/predict/route': ['models/parent_predictor/**']
    }
  },
  webpack: (config, { isServer }) => {
    if (isServer) {
      // Avoid bundling native tfjs-node in Next server build; require at runtime instead
      config.externals = config.externals || []
      config.externals.push('@tensorflow/tfjs-node')
    } else {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
      };
    }
    return config;
  },
};

module.exports = nextConfig;
