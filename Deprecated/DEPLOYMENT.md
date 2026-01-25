# 🚀 Deployment Guide for Ohana AI

## Quick Deployment Steps

### 1. Set up Database (Required)

Choose one of these options:

#### Option A: Neon (Recommended - Free tier available)
1. Go to [neon.tech](https://neon.tech)
2. Create a free account
3. Create a new project
4. Copy the connection string

#### Option B: Supabase
1. Go to [supabase.com](https://supabase.com)
2. Create a project
3. Go to Settings > Database
4. Copy the connection string

#### Option C: Vercel Postgres
1. In your Vercel dashboard
2. Go to Storage tab
3. Create Postgres database
4. Copy connection string

### 2. Deploy to Vercel

```bash
# Make sure you're in the project directory
cd /Users/tylergee/Documents/OhanaAI

# Deploy to Vercel
npx vercel --prod
```

### 3. Set Environment Variables in Vercel

Go to your Vercel dashboard → Project → Settings → Environment Variables

Add these variables:

```bash
# Database
DATABASE_URL="your-postgres-connection-string"

# Authentication (generate random strings)
NEXTAUTH_SECRET="your-random-secret-key"
NEXTAUTH_URL="https://your-app.vercel.app"

# ML Training API
ML_EXPORT_API_KEY="your-ml-api-key"
EXPORT_SECRET="your-export-secret"
```

### 4. Generate Secure Keys

Run these commands to generate secure keys:

```bash
# Generate NEXTAUTH_SECRET
openssl rand -base64 32

# Generate ML_EXPORT_API_KEY
openssl rand -base64 32

# Generate EXPORT_SECRET
openssl rand -base64 32
```

### 5. Update Training Config

After deployment, update `training_config.json`:

```json
{
  "web_app_url": "https://your-app.vercel.app",
  "api_key": "your-ml-export-api-key"
}
```

## Adding Your Images

### 1. Prepare Your Images

Create these images and place them in `public/images/`:

- `hero-family-tree.jpg` - Main hero image (1200x800px recommended)
- `upload-step.jpg` - Step 1 illustration (800x450px)
- `ai-analysis.jpg` - Step 2 illustration (800x450px) 
- `family-tree-result.jpg` - Step 3 illustration (800x450px)

### 2. Enable Images in Components

Uncomment the image components in:

- `components/ui/Hero.tsx` (line 73-79)
- `components/ui/Features.tsx` (line 123-128)

### 3. Optimize Images

```bash
# If you have ImageOptim or similar tools
# Compress images to reduce load times
```

## Post-Deployment Setup

### 1. Test the Application

1. Visit your deployed URL
2. Try signing up for an account
3. Upload a small GEDCOM file
4. Verify the family tree visualization works

### 2. Set up Database Schema

The database schema will be created automatically when you first run the app, but you can also run:

```bash
# If you want to manually set up the database
npx drizzle-kit push:pg
```

### 3. Configure ML Training Pipeline

On your local M1 Mac:

```bash
# Set up the ML environment
./setup_ml_environment.sh

# Update the config with your production URL
# Edit training_config.json with your Vercel URL

# Test the data pipeline
python scripts/auto_train.py --manual
```

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Verify DATABASE_URL is correct
   - Check if database allows connections from Vercel IPs

2. **Build Errors**
   - Make sure all dependencies are in package.json
   - Check for TypeScript errors

3. **Images Not Loading**
   - Verify images are in `public/images/` folder
   - Check image paths in components
   - Ensure images are properly sized

### Performance Optimization

1. **Image Optimization**
   ```bash
   # Add to next.config.js
   images: {
     domains: ['your-domain.com'],
     formats: ['image/webp', 'image/avif'],
   }
   ```

2. **Database Optimization**
   - Set up connection pooling
   - Add database indexes for frequently queried fields

3. **Caching**
   - Enable Vercel Edge Caching
   - Use Next.js ISR for static content

## Security Checklist

- [ ] Database connection uses SSL
- [ ] Environment variables are properly set
- [ ] API endpoints have rate limiting
- [ ] File uploads have size limits
- [ ] User data is properly sanitized

## Monitoring

### Set up Monitoring (Optional)

1. **Vercel Analytics**
   - Enable in Vercel dashboard
   - Monitor page performance

2. **Database Monitoring**
   - Set up alerts for connection issues
   - Monitor query performance

3. **ML Pipeline Monitoring**
   - Set up notifications for training completion
   - Monitor model performance metrics

## Scaling Considerations

As your app grows:

1. **Database Scaling**
   - Consider read replicas
   - Implement proper indexing

2. **File Storage**
   - Move large files to AWS S3 or similar
   - Implement CDN for static assets

3. **ML Pipeline**
   - Consider GPU instances for training
   - Implement model versioning

---

Your Ohana AI application should now be live and ready to help families discover their connections! 🌟