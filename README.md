# Ohana AI - Family Tree Intelligence

A web application that uses AI to predict missing family relationships in GEDCOM files. Built with Next.js, TypeScript, and TensorFlow.js, featuring Graph Neural Networks (GNN) and Graph Attention Networks (GAT) for parent prediction.

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
- **ML**: TensorFlow.js, Python training pipeline
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
   
   Edit \`.env.local\` with your database URL and other configuration.

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
   curl -X POST http://localhost:3000/api/ml/export-training-data \\
     -H "Content-Type: application/json" \\
     -d '{"authorization": "your-export-secret"}'
   \`\`\`

3. **Train the Model**
   \`\`\`bash
   cd training_data
   pip install -r requirements.txt
   python run_training.py
   \`\`\`

4. **Deploy the Model**
   The trained model is automatically saved to \`models/parent_predictor/\` and will be loaded by the web application.

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
- **Inference**: Real-time parent prediction via TensorFlow.js

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

## Deployment

### Vercel Deployment

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