import { NextRequest, NextResponse } from 'next/server'
import bcrypt from 'bcryptjs'
import { db } from '@/lib/db'
import { users } from '@/lib/db/schema'
import { eq } from 'drizzle-orm'
import { z } from 'zod'
import { sendEmail, formatAdminEmail } from '@/lib/email/mailer'

const registerSchema = z.object({
  name: z.string().min(1, 'Name is required'),
  email: z.string().email('Invalid email address'),
  password: z.string().min(6, 'Password must be at least 6 characters'),
})

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { name, email, password } = registerSchema.parse(body)

    const existingUser = await db.select().from(users).where(eq(users.email, email)).limit(1)
    
    if (existingUser.length > 0) {
      return NextResponse.json(
        { error: 'User with this email already exists' },
        { status: 400 }
      )
    }

    const hashedPassword = await bcrypt.hash(password, 12)

    const newUser = await db.insert(users).values({
      name,
      email,
      password: hashedPassword,
    }).returning({ id: users.id, name: users.name, email: users.email })

    // Fire-and-forget notifications
    const admin = process.env.ADMIN_EMAIL
    if (admin) {
      const payload = formatAdminEmail('New user joined', [
        `Name: ${name}`,
        `Email: ${email}`,
        `Time: ${new Date().toISOString()}`,
      ])
      sendEmail({ to: admin, ...payload }).catch(() => {})
    }
    // Welcome email
    try {
      await sendEmail({
        to: email,
        subject: 'Welcome to Ohana AI',
        text: `Hi ${name}, your account has been created.`,
      })
    } catch {}

    return NextResponse.json(
      { user: newUser[0] },
      { status: 201 }
    )

  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: error.errors[0].message },
        { status: 400 }
      )
    }

    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}
