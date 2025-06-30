#!/bin/bash
# Script to help you add and optimize images for Ohana AI

echo "🖼️  Ohana AI Image Setup Helper"
echo "================================"

# Create images directory if it doesn't exist
mkdir -p public/images

echo "📁 Created public/images directory"

# List what images are needed
echo ""
echo "📋 Images needed for your Ohana AI app:"
echo ""
echo "1. Hero Image (Main landing page)"
echo "   • File: public/images/hero-family-tree.jpg"
echo "   • Size: 1200x800px (3:2 aspect ratio)"
echo "   • Description: A beautiful family tree or genealogy visualization"
echo ""
echo "2. Upload Step Image"
echo "   • File: public/images/upload-step.jpg" 
echo "   • Size: 800x450px (16:9 aspect ratio)"
echo "   • Description: Someone uploading a file or GEDCOM visualization"
echo ""
echo "3. AI Analysis Image"
echo "   • File: public/images/ai-analysis.jpg"
echo "   • Size: 800x450px (16:9 aspect ratio)"
echo "   • Description: AI/ML visualization, network graphs, or data processing"
echo ""
echo "4. Family Tree Result Image"
echo "   • File: public/images/family-tree-result.jpg"
echo "   • Size: 800x450px (16:9 aspect ratio)"
echo "   • Description: Interactive family tree with connections highlighted"
echo ""

# Check if any images already exist
echo "🔍 Checking for existing images..."
images_found=0

if [ -f "public/images/hero-family-tree.jpg" ]; then
    echo "✅ Hero image found: hero-family-tree.jpg"
    images_found=$((images_found + 1))
else
    echo "❌ Missing: hero-family-tree.jpg"
fi

if [ -f "public/images/upload-step.jpg" ]; then
    echo "✅ Upload step image found: upload-step.jpg"
    images_found=$((images_found + 1))
else
    echo "❌ Missing: upload-step.jpg"
fi

if [ -f "public/images/ai-analysis.jpg" ]; then
    echo "✅ AI analysis image found: ai-analysis.jpg"
    images_found=$((images_found + 1))
else
    echo "❌ Missing: ai-analysis.jpg"
fi

if [ -f "public/images/family-tree-result.jpg" ]; then
    echo "✅ Family tree result image found: family-tree-result.jpg"
    images_found=$((images_found + 1))
else
    echo "❌ Missing: family-tree-result.jpg"
fi

echo ""
echo "📊 Images found: $images_found/4"

if [ $images_found -eq 4 ]; then
    echo ""
    echo "🎉 All images found! Running enablement script..."
    
    # Enable images in Hero component
    if grep -q "{\*/\*" components/ui/Hero.tsx; then
        echo "🔧 Enabling hero image in Hero component..."
        sed -i '' 's/{\/\* \(<Image\)/\1/g; s/\/> \*\/}/\/>/g' components/ui/Hero.tsx
        echo "✅ Hero image enabled"
    else
        echo "ℹ️  Hero image already enabled"
    fi
    
    # Enable images in Features component
    if grep -q "{\*/\*" components/ui/Features.tsx; then
        echo "🔧 Enabling step images in Features component..."
        sed -i '' 's/{\/\* \(<Image\)/\1/g; s/\/> \*\/}/\/>/g' components/ui/Features.tsx
        echo "✅ Step images enabled"
    else
        echo "ℹ️  Step images already enabled"
    fi
    
    echo ""
    echo "🚀 All images are now enabled! Your site will look amazing."
    echo "💡 Run 'npm run dev' to see your images in action."
    
else
    echo ""
    echo "📝 Next steps:"
    echo "1. Add your images to the public/images/ folder"
    echo "2. Name them exactly as shown above"
    echo "3. Run this script again to auto-enable them"
    echo ""
    echo "💡 Tips for great images:"
    echo "• Use high-quality family photos or genealogy visualizations"
    echo "• Ensure images are web-optimized (under 500KB each)"
    echo "• Consider using tools like TinyPNG to compress them"
    echo "• JPG format works best for photos, PNG for graphics with transparency"
fi

echo ""
echo "🎨 Image Ideas:"
echo "• Family reunion photos"
echo "• Historical family portraits"
echo "• Hand-drawn family trees"
echo "• Screenshots of genealogy software"
echo "• AI/tech visualizations from Unsplash"
echo ""
echo "📱 Don't forget: Test on mobile devices too!"

# Offer to open the images folder
if command -v open &> /dev/null; then
    echo ""
    read -p "🗂️  Open the images folder now? (y/n): " open_folder
    if [[ $open_folder =~ ^[Yy]$ ]]; then
        open public/images
        echo "📂 Opened public/images folder"
    fi
fi

echo ""
echo "✨ Happy designing! Your family tree app will look fantastic."