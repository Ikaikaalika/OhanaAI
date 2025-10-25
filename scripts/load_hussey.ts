import { config as loadEnv } from 'dotenv'
import { readFile } from 'fs/promises'
import { createHash } from 'crypto'
import bcrypt from 'bcryptjs'
import { v4 as uuidv4 } from 'uuid'
import { db, client as dbClient } from '@/lib/db'
import { users, gedcomFiles } from '@/lib/db/schema'
import { eq } from 'drizzle-orm'
import { parseGedcom } from '@/lib/gedcom/parser'
import { processGedcomImmediately } from '@/lib/jobs/gedcomProcessor'
import { exportPendingTrainingData } from '@/lib/ml/exporter'

async function ensureUser() {
  const email = process.env.SEED_USER_EMAIL || 'demo@ohana.local'
  const password = process.env.SEED_USER_PASSWORD || 'password123'
  const name = process.env.SEED_USER_NAME || 'Demo User'

  const existing = await db.select().from(users).where(eq(users.email, email)).limit(1)
  if (existing.length) {
    return existing[0]
  }

  const hashed = await bcrypt.hash(password, 12)
  const [user] = await db
    .insert(users)
    .values({
      email,
      password: hashed,
      name
    })
    .returning()
  console.log(`Created seed user ${email} / ${password}`)
  return user
}

async function main() {
  const user = await ensureUser()
  const gedcomPath = process.env.GEDCOM_PATH || 'Hussey Ohana.ged.txt'
  const raw = await readFile(gedcomPath, 'utf8')
  const parsedData = parseGedcom(raw)
  console.log(`Parsed ${parsedData.individuals?.length ?? 0} individuals, ${parsedData.families?.length ?? 0} families`)
  const buffer = Buffer.from(raw, 'utf8')
  const fileHash = createHash('sha256').update(buffer).digest('hex')

  const filename = `${uuidv4()}.ged`
  const [record] = await db
    .insert(gedcomFiles)
    .values({
      userId: user.id,
      filename,
      originalName: gedcomPath,
      fileSize: buffer.byteLength,
      fileHash,
      parsedData,
      isProcessed: false
    })
    .returning()

  console.log(`Inserted GEDCOM ${record.id}, processing...`)
  await processGedcomImmediately({ fileId: record.id, parsedData })
  console.log('Processing complete.')

  const exportResult = await exportPendingTrainingData()
  console.log(`Exported ${exportResult.count} training rows to ${exportResult.directory}`)
  await dbClient.end({ timeout: 0 })
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
loadEnv({ path: '.env.local' })
