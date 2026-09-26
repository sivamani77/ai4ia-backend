#!/usr/bin/env python3
"""Mint k6/loadtest_users.json: real accounts, real access + refresh tokens.

k6.js needs one real, signed-in user per VU (k6/loadtest_users.json — never
committed, see .gitignore). This script is what actually produces that file:

  1. Log in as an existing admin account.
  2. For each loadtest_NNN account, create it via POST /admin/users if it
     doesn't exist yet (admin-created accounts skip OTP/email verification —
     see admin_user_service.create_user — so this is idempotent: already-
     existing accounts just 409 and get reused).
  3. Log in as each loadtest_NNN account to mint a FRESH access_token
     (15-min TTL by default) and capture its refresh_token (an httpOnly
     Set-Cookie, not in the JSON body — see auth.py's _set_refresh_cookie).
  4. Write [{username, user_id, access_token, refresh_token}, ...] to the
     output file.

Run this right before a load test, not once — access tokens are short-lived,
and re-running is what "regenerate loadtest_users.json right before running"
in k6.js's own header comment means. Account creation (step 2) is safe to
re-run too; existing accounts are detected and reused, not recreated.

Usage:
  python3 k6/mint_loadtest_users.py --count 50 \
      --admin-username admin-001 --admin-password '...' \
      --base-url https://uat-api.ai4ia.io

Credentials are read from --admin-password or the LOADTEST_ADMIN_PASSWORD
env var — never hardcode them here or commit them anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

DEFAULT_BASE_URL = "https://uat-api.ai4ia.io"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "loadtest_users.json"
# Same fixed password for every loadtest account — they're synthetic,
# disposable, and never hold anything worth protecting individually.
LOADTEST_PASSWORD = "LoadTest#2026!"


def api(base_url: str) -> str:
    return base_url.rstrip("/") + "/api"


def login(base_url: str, username: str, password: str) -> dict:
    """POST /auth/login. Returns {access_token, user_id, refresh_token}.
    refresh_token comes from the Set-Cookie header, not the JSON body."""
    resp = requests.post(
        f"{api(base_url)}/auth/login",
        json={"username": username, "password": password},
        timeout=20,
    )
    resp.raise_for_status()
    body = resp.json()
    refresh_token = resp.cookies.get("refresh_token")
    return {
        "username": username,
        "user_id": body["user"]["id"],
        "access_token": body["access_token"],
        "refresh_token": refresh_token,
    }


def ensure_account(base_url: str, admin_token: str, username: str, email: str) -> bool:
    """POST /admin/users. A 409 (already exists) is expected and fine —
    idempotent by design, not an error. Returns True if newly created."""
    resp = requests.post(
        f"{api(base_url)}/admin/users",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "username": username,
            "password": LOADTEST_PASSWORD,
            "first_name": "Load",
            "last_name": username,
            "email": email,
            "persona": "general",
        },
        timeout=20,
    )
    if resp.status_code == 409:
        return False
    resp.raise_for_status()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=50, help="number of loadtest users (default 50)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--prefix", default="loadtest")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--admin-username", required=True)
    parser.add_argument(
        "--admin-password",
        default=os.environ.get("LOADTEST_ADMIN_PASSWORD"),
        help="or set LOADTEST_ADMIN_PASSWORD",
    )
    parser.add_argument(
        "--skip-create", action="store_true",
        help="skip account creation — only log in existing loadtest_NNN accounts to mint fresh tokens",
    )
    args = parser.parse_args()

    if not args.admin_password:
        print("error: --admin-password or LOADTEST_ADMIN_PASSWORD is required", file=sys.stderr)
        return 1

    print(f"Logging in as admin ({args.admin_username}) against {args.base_url}...")
    admin = login(args.base_url, args.admin_username, args.admin_password)
    admin_token = admin["access_token"]

    created = 0
    reused = 0
    results = []
    for i in range(1, args.count + 1):
        username = f"{args.prefix}_{i:03d}"
        # example.com (RFC 2606) — passes pydantic's EmailStr format check;
        # ".invalid" does not (rejected as a special-use TLD). No email is
        # ever sent to these addresses — admin-created accounts skip OTP.
        email = f"{username}@example.com"

        if not args.skip_create:
            if ensure_account(args.base_url, admin_token, username, email):
                created += 1
            else:
                reused += 1

        user = login(args.base_url, username, LOADTEST_PASSWORD)
        results.append(user)
        print(f"  [{i}/{args.count}] {username} -> token minted (user_id={user['user_id']})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} users to {args.output}")
    if not args.skip_create:
        print(f"  accounts created: {created}, reused: {reused}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
