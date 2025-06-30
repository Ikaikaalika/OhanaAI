#!/bin/bash
# Quick deployment script for Ohana AI

set -e  # Exit on any error

echo "🚀 Ohana AI Deployment Script"
echo "============================="

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found. Please run this script from the project root."
    exit 1
fi

# Check for required files
echo "🔍 Checking project files..."
required_files=("package.json" "next.config.js" "tailwind.config.ts" "app/layout.tsx")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing required file: $file"
        exit 1
    fi
done
echo "✅ All required files present"

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "📦 Installing Vercel CLI..."
    npm install -g vercel
fi

# Generate secure keys if .env.example exists but .env.local doesn't
if [ -f ".env.example" ] && [ ! -f ".env.local" ]; then
    echo "🔐 Generating secure environment variables..."
    
    # Generate random keys
    NEXTAUTH_SECRET=$(openssl rand -base64 32)
    ML_EXPORT_API_KEY=$(openssl rand -base64 32)
    EXPORT_SECRET=$(openssl rand -base64 32)
    
    echo "✅ Generated secure keys"
    echo ""
    echo "📋 Environment Variables for Vercel:"
    echo "===================================="
    echo "Copy these to your Vercel dashboard → Settings → Environment Variables:"
    echo ""
    echo "NEXTAUTH_SECRET=$NEXTAUTH_SECRET"
    echo "ML_EXPORT_API_KEY=$ML_EXPORT_API_KEY"
    echo "EXPORT_SECRET=$EXPORT_SECRET"
    echo ""
    echo "⚠️  You'll also need to add:"
    echo "DATABASE_URL=your-postgres-connection-string"
    echo "NEXTAUTH_URL=https://your-app.vercel.app"
    echo ""
    
    # Save to local file for reference
    cat > .env.local << EOF
# Generated environment variables for Ohana AI
# These are also displayed above for Vercel deployment

NEXTAUTH_SECRET=$NEXTAUTH_SECRET
ML_EXPORT_API_KEY=$ML_EXPORT_API_KEY
EXPORT_SECRET=$EXPORT_SECRET

# Add these manually:
# DATABASE_URL=your-postgres-connection-string
# NEXTAUTH_URL=https://your-app.vercel.app
EOF
    
    echo "💾 Saved keys to .env.local for local development"
fi

# Build and test locally first
echo ""
echo "🔨 Building project locally to check for errors..."
npm run build

if [ $? -eq 0 ]; then
    echo "✅ Local build successful"
else
    echo "❌ Local build failed. Please fix errors before deploying."
    exit 1
fi

# Check for images
echo ""
echo "🖼️  Checking for images..."
if [ -d "public/images" ] && [ "$(ls -A public/images)" ]; then
    echo "✅ Images folder found with content"
    ls -la public/images/
else
    echo "⚠️  No images found in public/images/"
    echo "💡 Run './add_images.sh' to set up your images"
fi

# Ask about database setup
echo ""
echo "💾 Database Setup"
echo "================="
echo "Have you set up your PostgreSQL database? (Neon, Supabase, or Vercel Postgres)"
echo ""
echo "Database providers:"
echo "• Neon: https://neon.tech (recommended)"
echo "• Supabase: https://supabase.com"
echo "• Vercel Postgres: In your Vercel dashboard"
echo ""
read -p "Do you have your DATABASE_URL ready? (y/n): " db_ready

if [[ ! $db_ready =~ ^[Yy]$ ]]; then
    echo ""
    echo "⏸️  Deployment paused. Please:"
    echo "1. Set up a PostgreSQL database"
    echo "2. Get your DATABASE_URL connection string"
    echo "3. Run this script again"
    echo ""
    echo "🔗 Quick setup links:"
    echo "• Neon: https://console.neon.tech/signup"
    echo "• Supabase: https://app.supabase.com/"
    exit 0
fi

# Deploy to Vercel
echo ""
echo "🚀 Deploying to Vercel..."
echo "========================"

# Check if already linked to a Vercel project
if [ -d ".vercel" ]; then
    echo "📱 Project already linked to Vercel"
    vercel --prod
else
    echo "🔗 Linking to Vercel project..."
    vercel --prod
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Deployment successful!"
    echo ""
    echo "📋 Next Steps:"
    echo "=============="
    echo "1. 🌐 Visit your deployed app (URL shown above)"
    echo "2. ⚙️  Add environment variables in Vercel dashboard:"
    echo "   • Go to your project → Settings → Environment Variables"
    echo "   • Add the variables shown earlier in this script"
    echo "3. 🔄 Redeploy after adding environment variables"
    echo "4. 🧪 Test signup, login, and file upload"
    echo ""
    echo "🖼️  To add your images:"
    echo "   • Run: ./add_images.sh"
    echo "   • Add your family photos to public/images/"
    echo ""
    echo "🤖 To set up ML training:"
    echo "   • Run: ./setup_ml_environment.sh"
    echo "   • Update training_config.json with your app URL"
    echo ""
    echo "✨ Your AI-powered family tree app is live!"
    
else
    echo "❌ Deployment failed. Please check the error messages above."
    exit 1
fi