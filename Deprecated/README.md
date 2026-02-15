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
   pip install -r requirements_mlx.txt   # install MLX/ONNX tooling once
   python3 train_model_mlx.py --data-dir training_data --output-dir models/parent_predictor
   \`\`\`

4. **Deploy the Model**
   Training writes \`models/parent_predictor/model.onnx\`. The Next.js API loads this ONNX via `onnxruntime-node`; restart the server (or redeploy) after replacing the file.

#### Alternative: Automated Training

- `setup_ml_environment.sh` now installs MLX packages (instead of TensorFlow) and prepares the local virtualenv.
- `scripts/auto_train.py` can be scheduled to fetch exports and run `train_model_mlx.py` automatically.

### Training labels & attributes

Each `ml_training_data.labels` entry now stores a richer snapshot per person so future models can predict detailed facts:

- `name`, `gender`, `birthDate`, `birthPlace`, `deathDate`, `deathPlace`
- `residences`: array of `{ place, date }` pulled from GEDCOM `RESI` events
- `spouses`: array of `{ id, name, birthDate, birthPlace }` derived from family relationships
- `missingAttributes`: booleans indicating whether each attribute is currently unknown

This phase lays the groundwork for upcoming models that infer not just “is a parent missing?” but also the concrete names, dates, and locations needed to fill the tree.

### Model Files Checklist

Place the following under \`models/parent_predictor/\`:

- TF.js graph model: \`model.json\` + shard binaries (\`group*.bin\`)
- Optional metadata: \`training_metadata.json\`, \`training_history.json\`

Model resolution order at runtime:
1. \`TFJS_MODEL_URL\` env var (explicit override)
2. latest \`/api/ml/training-status\` row with \`status=ready|completed\` and \`details.tfjsModelUrl\`
3. local \`models/parent_predictor/model.json\`

### Continuous Training

Set up a cron job or GitHub Action to periodically:
1. Export new training data
2. Retrain locally on MLX
3. Convert/upload TF.js artifacts
4. POST \`/api/ml/training-status\` with \`status=ready\` and new \`modelVersion\`

When a \`ready/completed\` status is posted, the app automatically:
- Refreshes predictions for all processed GEDCOM files
- Updates stored \`gedcom_files.predictions\` and \`modelVersion\`
- Sends user update emails (unless \`SEND_MODEL_UPDATE_EMAILS=false\`)

### Mac Polling Agent (Vercel-Friendly)

When deployed on Vercel, run the local Mac agent to poll for training exports,
download them from Vercel Blob, retrain locally, and mark the data as trained.

1. **Set Vercel env vars**:
   - `ML_EXPORT_API_KEY` (shared secret)
   - `BLOB_READ_WRITE_TOKEN` (Vercel Blob token)

2. **Run the agent on your Mac**:
   ```bash
   cd /Users/tylergee/Documents/OhanaAI/Deprecated
   export OHANA_API_BASE="https://your-app.vercel.app"
   export ML_EXPORT_API_KEY="your-ml-export-api-key"
   python3 scripts/mac_agent.py --interval 300
   ```

3. **One-shot mode** (optional):
   ```bash
   python3 scripts/mac_agent.py --once
   ```

4. **TF.js model export (automatic)**:
   The agent converts the ONNX model to TF.js after every successful training
   and uploads the TF.js artifacts to Vercel Blob. Ensure the following tools
   are available in your Python environment:
   - `onnx-tf`
   - `tensorflowjs`
   - `tensorflow` (or `tensorflow-macos` + `tensorflow-metal`)

   If you need a custom conversion pipeline, you can set:
   - `ONNX_TO_TF_CMD` (default: `onnx-tf convert -i {onnx} -o {saved_model}`)
   - `TFJS_CONVERTER_CMD` (default: `tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model {saved_model} {tfjs_dir}`)

## Architecture

### Data Flow

1. **User creates account and uploads GEDCOM** → File is parsed, stored in DB, and saved to blob/local storage.
2. **Tree + ML examples generated** → \`family_trees\` and \`ml_training_data\` are created.
3. **Initial inference runs** → upload pipeline and \`/api/ml/predict\` generate/store parent predictions from TF.js model.
4. **Local training loop** → Mac agent polls \`/api/ml/export-user-data\`, downloads batches + source GEDCOM blob files, trains with MLX, converts to TF.js, uploads artifacts.
5. **Global refresh** → \`/api/ml/training-status\` triggers a refresh for all users and sends update emails.

### Database Schema

- \`users\`: User accounts and authentication
- \`gedcom_files\`: Uploaded files and metadata
- \`family_trees\`: Parsed family relationships
- \`ml_training_data\`: Processed data for model training

### ML Pipeline

- **Graph Construction**: Convert family trees to graph structures
- **Feature Engineering**: Extract per-individual feature vectors (12 dims)
- **Model Training**: MLX model trained locally and exported to ONNX
- **Model Serving**: ONNX converted to TF.js graph model and uploaded to blob via \`/api/ml/upload-tfjs\`
- **Inference**: Server-side parent prediction in \`/api/ml/predict\` via TensorFlow.js loader
  - Confidence filtering is controlled via `PREDICTION_CONFIDENCE_THRESHOLD` (default 0.4) and `PREDICTION_MAX_SUGGESTIONS` (optional).

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
- \`POST /api/ml/export-user-data\` - Export pending training rows (+ source GEDCOM blob pointers)
- \`POST /api/ml/upload-tfjs\` - Upload TF.js model artifacts
- \`POST /api/ml/training-status\` - Publish training progress/completion (triggers global refresh on ready/completed)
- \`POST /api/ml/mark-trained\` - Mark exported training IDs as included

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

   - Uploaded GEDCOM files are saved under `uploads/` (configurable via `STORAGE_ROOT`/`UPLOADS_DIR`). Each upload is hashed per-user so duplicate GEDCOMs are rejected with a helpful message.
   - Models are loaded from `models/parent_predictor/`
   - Training exports written to `training_data/` (configurable via `TRAINING_DATA_DIR`)

   Override the defaults in `.env.local`:

   ```
   STORAGE_ROOT=/Volumes/OhanaData     # optional absolute path (e.g., external drive)
   UPLOADS_DIR=uploads                 # relative to STORAGE_ROOT if not absolute
   TRAINING_DATA_DIR=training_data     # relative to STORAGE_ROOT if not absolute
   ML_EXPORTS_DIR=exports/ml_training  # where export-user-data writes files
   SYNC_GEDCOM_PROCESSING=true         # block upload response until parsing + inference complete
   MISSING_PARENT_THRESHOLD=0.5        # minimum threshold to infer a missing parent
   PREDICTION_CONFIDENCE_THRESHOLD=0.4 # minimum confidence for suggested parents
   PREDICTION_MAX_SUGGESTIONS=5        # cap suggestions per missing parent (set 0 for unlimited)
   TFJS_MODEL_URL=                     # optional explicit TF.js model.json URL
   TFJS_MODEL_VERSION=                 # optional explicit model version label
   SEND_MODEL_UPDATE_EMAILS=true       # notify users when global refresh runs
   ```

   If you leave these blank the server stores files alongside the app directory (or `/tmp/ohana-ai` on Vercel). All paths are created automatically at runtime.

5) GEDCOM processing

   - By default (`SYNC_GEDCOM_PROCESSING=true`) uploads stay open until parsing, ML data prep, and TF.js inference finish so users can wait for predictions.
   - Set the env to `false` to fall back to the asynchronous queue (`lib/jobs/gedcomProcessor.ts`).

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
