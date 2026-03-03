"""
config.py – Centralised settings loader.

All environment variables are read ONCE at startup. If a required variable is
missing the application refuses to start with a clear error message (fail-fast).

OWASP ref: https://owasp.org/www-project-application-security-verification-standard/
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    """Return the value of an env var, raising RuntimeError if absent/empty."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"[config] Required environment variable '{name}' is not set. "
            "Add it to your .env file and restart the server."
        )
    return value


def _optional(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


# ── Required secrets (crash early if missing) ─────────────────────────────────
SUPABASE_URL: str = _require("SUPABASE_URL")
SUPABASE_KEY: str = _require("SUPABASE_KEY")

# ── Optional / configurable ───────────────────────────────────────────────────

# Comma-separated list of allowed CORS origins.
# Example: ALLOWED_ORIGINS=https://app.example.com,https://www.example.com
# In production NEVER leave this as "*".
_raw_origins = _optional("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# Application environment: "development" shows /docs; "production" hides them.
APP_ENV: str = _optional("APP_ENV", "development")

# Global default rate-limit string parsed by slowapi (e.g. "60/minute").
RATE_LIMIT_DEFAULT: str = _optional("RATE_LIMIT_DEFAULT", "60/minute")

# Stricter limit for expensive endpoints (search, head-to-head).
RATE_LIMIT_SEARCH: str = _optional("RATE_LIMIT_SEARCH", "30/minute")
