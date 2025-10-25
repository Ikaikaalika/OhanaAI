# Domain Setup (Cloudflare + Vercel)

Use this guide to point `ohanaai.org` to the Vercel deployment and keep auth working across redeploys.

## Prerequisites
- Vercel project linked (see `.vercel/project.json`).
- Vercel CLI authenticated (`vercel whoami`).
- Access to Cloudflare zone for `ohanaai.org`.

## Cloudflare DNS
1) Add A record (apex)
- Type: A
- Name: @
- Content: 76.76.21.21
- Proxy status: DNS only (grey cloud)

2) Add CNAME record (www)
- Type: CNAME
- Name: www
- Target: cname.vercel-dns.com
- Proxy status: DNS only (grey cloud)

3) Remove conflicting records for `@`/`www` (A/AAAA/CNAME). In SSL/TLS, choose Full (strict). Optional: enable “Always Use HTTPS”.

## Vercel Configuration
- Add domains to the project:
  - `vercel domains add ohanaai.org`
  - `vercel domains add www.ohanaai.org`
  - If Vercel shows a TXT verification, add it in Cloudflare DNS and retry.

- Alias current production deployment to the domain:
  - `DEPLOY=$(vercel ls --confirm --yes | head -n1)`
  - `vercel alias set "$DEPLOY" ohanaai.org`

- Set `NEXTAUTH_URL` and redeploy (production):
  - `echo "https://ohanaai.org" | vercel env add NEXTAUTH_URL production`
  - `vercel --prod`

## Optional Redirects
- Cloudflare redirect (www → apex): Rules > Redirect Rules > Create
  - If Hostname equals `www.ohanaai.org` → Static redirect to `https://ohanaai.org` (301)
- Or Vercel config: add redirect in `vercel.json`.

## Verify
- DNS: `dig +short ohanaai.org A` → `76.76.21.21`
- DNS: `dig +short www.ohanaai.org CNAME` → `cname.vercel-dns.com`
- Open: `https://ohanaai.org`

## Troubleshooting
- 404: Domain not added/aliased on Vercel, or DNS not propagated.
- 401: Deployment protection enabled; adjust in Vercel Project/Team settings.
- SSL loops: Ensure Cloudflare proxy is OFF (DNS only) and SSL/TLS is Full (strict).
