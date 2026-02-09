#!/usr/bin/env python3
"""
Cloudflare DNS setup for Vercel (A records).

Wrangler is for Workers/Pages and does not manage DNS records. This script uses the
Cloudflare v4 API to upsert DNS records for a zone.

Default behavior matches the Vercel CLI guidance:
- A @ -> 76.76.21.21
- A www -> 76.76.21.21

Usage:
  export CLOUDFLARE_API_TOKEN="..."
  python3 scripts/cloudflare_dns_setup.py ohanaai.org
  python3 scripts/cloudflare_dns_setup.py ohanaai.com ohanaai.org --ip 76.76.21.21
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional
from urllib import error, request, parse


API_BASE = "https://api.cloudflare.com/client/v4"


def _api_request(token: str, method: str, path: str, body: Optional[dict] = None) -> dict:
    url = f"{API_BASE}{path}"
    data = None
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")

    req = request.Request(url, method=method, headers=headers, data=data)
    try:
        with request.urlopen(req, timeout=60) as resp:
            payload = resp.read().decode("utf-8")
    except error.HTTPError as e:
        details = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Cloudflare API error {e.code} {e.reason}: {details}") from e
    except Exception as e:
        raise RuntimeError(f"Cloudflare API request failed: {e}") from e

    try:
        parsed_payload = json.loads(payload) if payload else {}
    except Exception as e:
        raise RuntimeError(f"Cloudflare API returned invalid JSON: {payload[:2000]}") from e

    if not parsed_payload.get("success", False):
        errs = parsed_payload.get("errors") or []
        raise RuntimeError(f"Cloudflare API reported failure: {errs}")
    return parsed_payload


def _get_zone_id(token: str, zone_name: str) -> str:
    q = parse.urlencode({"name": zone_name})
    resp = _api_request(token, "GET", f"/zones?{q}")
    results = resp.get("result") or []
    if not results:
        raise RuntimeError(f"Zone not found in Cloudflare account: {zone_name}")
    # Prefer exact match
    for z in results:
        if (z.get("name") or "").lower() == zone_name.lower():
            return str(z["id"])
    return str(results[0]["id"])


def _find_dns_record(token: str, zone_id: str, record_type: str, name: str) -> Optional[dict]:
    q = parse.urlencode({"type": record_type, "name": name, "per_page": 100})
    resp = _api_request(token, "GET", f"/zones/{zone_id}/dns_records?{q}")
    results = resp.get("result") or []
    if not results:
        return None
    return results[0]


def _upsert_a_record(token: str, zone_id: str, name: str, ip: str, proxied: bool) -> dict:
    existing = _find_dns_record(token, zone_id, "A", name)
    body = {
        "type": "A",
        "name": name,
        "content": ip,
        "ttl": 1,  # auto
        "proxied": proxied,
    }

    if existing:
        rec_id = existing["id"]
        return _api_request(token, "PUT", f"/zones/{zone_id}/dns_records/{rec_id}", body).get("result")  # type: ignore[return-value]

    return _api_request(token, "POST", f"/zones/{zone_id}/dns_records", body).get("result")  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(description="Upsert Cloudflare DNS for Vercel")
    parser.add_argument("domains", nargs="+", help="Zone names (e.g. ohanaai.org)")
    parser.add_argument("--ip", default="76.76.21.21", help="Vercel A record IP (default: 76.76.21.21)")
    parser.add_argument("--no-www", action="store_true", help="Skip creating the www record")
    parser.add_argument("--proxied", action="store_true", help="Enable Cloudflare proxying (not recommended for Vercel)")
    parser.add_argument("--token-file", default=None, help="Path to a file containing CLOUDFLARE_API_TOKEN")
    args = parser.parse_args()

    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    if not token and args.token_file:
        try:
            with open(args.token_file, "r") as fh:
                token = fh.read().strip()
        except Exception as e:
            print(f"Failed to read token file: {e}", file=sys.stderr)
            return 2
    if not token:
        print("Missing CLOUDFLARE_API_TOKEN in environment", file=sys.stderr)
        return 2

    for domain in args.domains:
        zone_id = _get_zone_id(token, domain)
        print(f"[cf] zone={domain} zone_id={zone_id}")

        apex = domain
        print(f"[cf] upsert A {apex} -> {args.ip} proxied={args.proxied}")
        _upsert_a_record(token, zone_id, apex, args.ip, args.proxied)

        if not args.no_www:
            www = f"www.{domain}"
            print(f"[cf] upsert A {www} -> {args.ip} proxied={args.proxied}")
            _upsert_a_record(token, zone_id, www, args.ip, args.proxied)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
