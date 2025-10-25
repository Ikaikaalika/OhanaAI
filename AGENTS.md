# Repository Guidelines

## Project Structure & Module Organization
- `app/`: Next.js App Router pages, API routes (`app/api/**/route.ts`).
- `components/`: Reusable React components (PascalCase `.tsx`).
- `lib/`: Server/client utilities (DB, helpers), plus `drizzle.config.ts` at root.
- `types/`: Shared TypeScript types.
- `public/`: Static assets.
- `scripts/` and root ML files: ML automation (`scripts/auto_train.py`), setup (`setup_ml_environment.sh`), training (`train_model_m1.py`, `train_model.py`).
- `models/parent_predictor/`: Trained model artifacts loaded by predict API.
- `README.md`, `DEPLOYMENT.md`: High-level docs. Env samples in `.env.example`.

## Build, Test, and Development Commands
- `npm run dev`: Start local Next.js dev server (`http://localhost:3000`).
- `npm run build`: Production build; `npm start`: run the built app.
- `npm run lint`: Lint with Next/ESLint.
- Database (Drizzle): `npm run db:migrate` (generate), `npm run db:push` (apply), `npm run db:studio` (inspect).
- ML (per README):
  1) `curl -X POST /api/ml/export-training-data -d '{"authorization":"$EXPORT_SECRET"}'`
  2) `cd training_data && pip install -r requirements.txt`
  3) `python run_training.py` → saves to `models/parent_predictor/`
  Alternative: `bash setup_ml_environment.sh`, then `python3 train_model_m1.py` or `python3 train_model.py`.

## Coding Style & Naming Conventions
- TypeScript, 2‑space indent; prefer explicit types at module boundaries.
- React components: PascalCase in `components/`, hooks `use*.ts`, server utilities in `lib/`.
- App Router segments lowercase with dashes; API handlers in `app/api/**/route.ts`.
- Run `npm run lint` before commits; Tailwind utility-first styles kept readable and grouped logically.

## Testing Guidelines
- No formal test suite yet. When adding tests:
  - Unit: Jest + Testing Library (`__tests__/**`, `*.test.ts(x)`).
  - E2E: Playwright (`e2e/**`).
  - Cover critical flows: auth, GEDCOM import/export, predictions.

## Commit & Pull Request Guidelines
- Use Conventional Commits: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`. Example: `feat: add parent prediction endpoint`.
- PRs must include: concise description, linked issue, screenshots for UI, and steps to validate.
- Require green `npm run lint` and a successful local run (`npm run dev` or `npm start`).

## Security & Configuration
- Create env: `cp .env.example .env.local`; fill `DATABASE_URL`, `NEXTAUTH_SECRET`, `NEXTAUTH_URL`, `EXPORT_SECRET`, `ML_EXPORT_API_KEY`, `VERCEL_URL`.
- Protect secrets; do not commit. Place model artifacts under `models/parent_predictor/` (bundled for the predict API).
- Custom domains: see `DOMAIN_SETUP.md` for Cloudflare + Vercel steps.
