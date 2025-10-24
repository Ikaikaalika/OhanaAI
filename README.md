# Ohana AI - Family Tree Intelligence

A web application that uses AI to predict missing family relationships in GEDCOM files. Built with Next.js, TypeScript, and TensorFlow.js (server-side via tfjs-node), featuring Graph Neural Networks (GNN) and Graph Attention Networks (GAT) for parent prediction.

## Features

- **User Authentication**: Secure login and registration system
- **GEDCOM File Upload**: Support for standard genealogy file formats
- **Interactive Family Trees**: Visual representation with vis-network
- **AI-Powered Predictions**: Missing parent prediction using GNN/GAT models
- **Data Privacy**: Users control their data with full deletion capabilities
- **Model Training Pipeline**: Continuous learning from user data
- **Export Capabilities**: Download enhanced family tree data

## Tech Stack

- **Frontend**: Next.js 14, TypeScript, Tailwind CSS
- **Backend**: Next.js API Routes, NextAuth.js
- **Database**: PostgreSQL with Drizzle ORM
- **ML**: TensorFlow.js (tfjs-node on the server), Python training pipeline
- **Visualization**: vis-network for family trees
- **Deployment**: Vercel

## Quick Start

### Prerequisites

- Node.js 18+
- PostgreSQL database
- Python 3.8+ (for ML training)

### Installation

1. **Clone the repository**
   \`\`\`bash
   git clone <your-repo-url>
   cd OhanaAI
   \`\`\`

2. **Install dependencies**
   \`\`\`bash
   npm install
   \`\`\`

3. **Set up environment variables**
   \`\`\`bash
   cp .env.example .env.local
   \`\`\`
   
   Edit \`.env.local\` with at least:
   - `DATABASE_URL`
   - `NEXTAUTH_SECRET`
   - `NEXTAUTH_URL` (e.g., http://localhost:3000)
   - `EXPORT_SECRET` (used by ML export endpoint)
   - `ML_EXPORT_API_KEY` (for training scripts that reference it)

4. **Set up the database**
   \`\`\`bash
   npm run db:migrate
   npm run db:push
   \`\`\`

5. **Start development server**
   \`\`\`bash
   npm run dev
   \`\`\`

   Visit [http://localhost:3000](http://localhost:3000)

## ML Training Pipeline

### Initial Setup (No Model Available)

When first deployed, the application will show "No trained model available" for predictions. To train your first model:

1. **Collect Training Data**
   - Users upload GEDCOM files through the web interface
   - Data is automatically processed and prepared for training

2. **Export Training Data**
   \`\`\`bash
   # Ensure EXPORT_SECRET is set in .env.local and server is running
   curl -X POST http://localhost:3000/api/ml/export-training-data \\
     -H "Content-Type: application/json" \\
     -d "{\"authorization\": \"$EXPORT_SECRET\"}"
   \`\`\`

   Validate export (example response):
   \`\`\`json
   {
     "message": "Training data exported successfully",
     "count": 123,
     "batches": 2,
     "directory": "/abs/path/to/training_data"
   }
   \`\`\`

3. **Train the Model**
   \`\`\`bash
   cd training_data
   pip install -r requirements.txt
   python3 run_training.py
   \`\`\`

4. **Deploy the Model**
   The trained model is saved to \`models/parent_predictor/\`. These artifacts are bundled for the serverless/API runtime and loaded by `/api/ml/predict` (see `next.config.js` `outputFileTracingIncludes`).

#### Alternative: Local Training Scripts

For local iterative training (especially on Apple Silicon):

\`\`\`bash
bash setup_ml_environment.sh   # sets up Python venv with TensorFlow
python3 train_model_m1.py      # Apple Silicon optimized
# or
python3 train_model.py
\`\`\`

### Model Files Checklist

Place the following under \`models/parent_predictor/\`:

- Keras model: \`model.h5\` or \`ohana_model.h5\` (or Apple Silicon: \`ohana_model_m1.h5\`).
- Optional TFJS artifacts (if converted): \`model.json\` and \`group1-shard1.bin\` (and additional shards if split).
- Optional metadata: \`training_metadata.json\`, \`training_history.png\`.

After placing files, restart the server. The predict API will load artifacts from this directory.

### Continuous Training

Set up a cron job or GitHub Action to periodically:
1. Export new training data
2. Retrain the model with updated data
3. Deploy the improved model

## Architecture

### Data Flow

1. **User uploads GEDCOM** → Parsed and stored in database
2. **Family tree created** → Relationships extracted and visualized
3. **ML data prepared** → Graph structure created for training
4. **Model inference** → Predictions generated for missing parents
5. **Results displayed** → Interactive family tree with predictions

### Database Schema

- \`users\`: User accounts and authentication
- \`gedcom_files\`: Uploaded files and metadata
- \`family_trees\`: Parsed family relationships
- \`ml_training_data\`: Processed data for model training

### ML Pipeline

- **Graph Construction**: Convert family trees to graph structures
- **Feature Engineering**: Extract person and relationship features
- **Model Training**: GNN/GAT models for link prediction
- **Inference**: Server-side parent prediction via TensorFlow.js (tfjs-node) in `/api/ml/predict`

## API Endpoints

### Authentication
- \`POST /api/auth/register\` - User registration
- \`POST /api/auth/[...nextauth]\` - NextAuth.js endpoints

### GEDCOM Management
- \`POST /api/gedcom/upload\` - Upload GEDCOM file
- \`GET /api/gedcom/[id]\` - Get file details
- \`DELETE /api/gedcom/[id]\` - Delete file and all data

### ML Operations
- \`POST /api/ml/predict\` - Generate parent predictions
- \`POST /api/ml/export-training-data\` - Export training data

## Local Hosting

Run locally with all data, models, and emails hosted on this device:

1) Environment

   - Copy env: `cp .env.example .env.local`
   - Set `DATABASE_URL` to your local Postgres
   - Set `NEXTAUTH_SECRET` and `NEXTAUTH_URL` (e.g., http://ohana.local or http://localhost:3000)
   - Optionally set email SMTP: `EMAIL_HOST`, `EMAIL_PORT`, `EMAIL_USER`, `EMAIL_PASS`, `EMAIL_FROM`, `ADMIN_EMAIL`
   - Optional auto-train: `AUTO_TRAIN=true`, `TRAINING_SCRIPT=train_model_m1.py`

2) Start app

   - `npm run dev` (or `npm run build && npm start`)
   - Optional: add `ohana.local` to `/etc/hosts` and front with a local proxy (Caddy/Nginx) for a custom URL.

3) Data storage

   - Uploaded GEDCOM files are saved under `uploads/` (configurable via `STORAGE_ROOT`/`UPLOADS_DIR`)
   - Models are loaded from `models/parent_predictor/`
   - Training exports written to `training_data/` (configurable via `TRAINING_DATA_DIR`)

   Override the defaults in `.env.local`:

   ```
   STORAGE_ROOT=/Volumes/OhanaData     # optional absolute path (e.g., external drive)
   UPLOADS_DIR=uploads                 # relative to STORAGE_ROOT if not absolute
   TRAINING_DATA_DIR=training_data     # relative to STORAGE_ROOT if not absolute
   ML_EXPORTS_DIR=exports/ml_training  # where export-user-data writes files
   ```

   If you leave these blank the server stores files alongside the app directory (or `/tmp/ohana-ai` on Vercel). All paths are created automatically at runtime.

5) GEDCOM processing queue

   - Uploads are parsed and converted to ML-ready data by an in-process background queue.
   - The API responds as soon as the file is stored; the queue continues building the family tree, so keep the Node.js server running until processing completes (`FileList` will show status).

4) Notifications (optional)

   - On user signup, account deletion, and new parent predictions, an email is sent to `ADMIN_EMAIL` (and a welcome email to the user) if SMTP is configured.

## Deployment

### Vercel Deployment

For custom domains via Cloudflare, see DOMAIN_SETUP.md.

1. **Connect to Vercel**
   \`\`\`bash
   npx vercel
   \`\`\`

2. **Set Environment Variables**
   - Add database URL, NextAuth secret, etc.

3. **Deploy**
   \`\`\`bash
   npx vercel --prod
   \`\`\`

### Database Setup

1. **Create PostgreSQL database** (recommended: Neon, Supabase, or Vercel Postgres)
2. **Run migrations**
3. **Update environment variables**

## Privacy & Security

- **Data Encryption**: All sensitive data is encrypted
- **User Control**: Complete data deletion capabilities
- **Access Control**: Users can only access their own data
- **Secure Authentication**: NextAuth.js with secure sessions

## Development

### Database Operations
\`\`\`bash
npm run db:studio     # Open Drizzle Studio
npm run db:migrate    # Generate migrations
npm run db:push       # Push schema changes
\`\`\`

### Testing
\`\`\`bash
npm run lint          # ESLint
npm run build         # Production build
\`\`\`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Open a GitHub issue
- Check the documentation
- Review the API endpoints

---

**Note**: This application is designed for genealogical research and family history. All predictions should be verified through traditional genealogical methods.
