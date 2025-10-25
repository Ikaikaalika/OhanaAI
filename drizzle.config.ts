import type { Config } from 'drizzle-kit'

export default {
  schema: './lib/db/schema.ts',
  out: './drizzle',
  driver: 'pg',
  tsconfig: './tsconfig.drizzle.json',
  dbCredentials: {
    connectionString: process.env.DATABASE_URL!,
  },
} satisfies Config
