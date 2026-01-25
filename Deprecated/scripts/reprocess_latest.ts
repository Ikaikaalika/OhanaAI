import { config as loadEnv } from 'dotenv'
import { db, client as dbClient } from '@/lib/db'
import { gedcomFiles } from '@/lib/db/schema'
import { desc } from 'drizzle-orm'
import { processGedcomImmediately } from '@/lib/jobs/gedcomProcessor'
import { ParsedGedcom } from '@/lib/gedcom/parser'

loadEnv({ path: '.env.local' })

async function main() {
  const rows = await db.select().from(gedcomFiles).orderBy(desc(gedcomFiles.processedAt)).limit(1)
  if (!rows.length) {
    throw new Error('No GEDCOM files found in database')
  }
  const file = rows[0]
  const parsedData = file.parsedData as ParsedGedcom
  console.log(`Reprocessing ${file.id} (${file.originalName})`)
  await processGedcomImmediately({ fileId: file.id, parsedData })
  console.log('Reprocessing complete.')
  await dbClient.end({ timeout: 0 })
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
