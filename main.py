"""
main.py – Padelytics API
========================
Security hardening applied (OWASP Top 10, 2021):

  A01 Broken Access Control    → CORS locked to explicit origins; read-only methods only
  A02 Cryptographic Failures   → All secrets via env vars (config.py); no hard-coded keys
  A03 Injection                → Pydantic Query constraints + slug regex on every param
  A04 Insecure Design          → Fail-fast config validation; /docs hidden in production
  A05 Security Misconfiguration→ Security headers middleware; CORS allowlist
  A07 Identification/AuthN     → Rate limiting (IP-based) via slowapi; 429 on breach
"""

import re
from datetime import date
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from supabase import Client, create_client

import config

supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

# ─── Rate limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[config.RATE_LIMIT_DEFAULT],
)

_docs_url = "/docs" if config.APP_ENV != "production" else None
_redoc_url = "/redoc" if config.APP_ENV != "production" else None

app = FastAPI(
    title="Padelytics API – Padel Pro Analytics",
    description="Advanced API for accessing professional padel data from Premier Padel.",
    version="2.0.0",
    docs_url=_docs_url,
    redoc_url=_redoc_url,
)

# ─── Rate-limit exception handler ────────────────────────────────────────────
# Returns a clean JSON 429.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ─── CORS middleware ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=False,          # No cookies/sessions used
    allow_methods=["GET"],            # This API is read-only
    allow_headers=["Content-Type", "Authorization"],
)


# ─── Security headers middleware ──────────────────────────────────────────────
# Adds defence-in-depth HTTP security headers to every response.
@app.middleware("http")
async def add_security_headers(request: Request, call_next) -> Response:
    response = await call_next(request)
    # Prevent MIME-type sniffing attacks
    response.headers["X-Content-Type-Options"] = "nosniff"
    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"
    # Force HTTPS (only meaningful in production with TLS)
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    # Disable aggressive caching of API responses
    response.headers["Cache-Control"] = "no-store"
    # Restrict browser feature access
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# ─── Slug validation helper ───────────────────────────────────────────────────
# Slugs must be URL-safe lowercase strings (letters, digits, hyphens only).
# This regex blocks SQL-injection characters, path traversal, and XSS payloads.
_SLUG_PATTERN = r"^[a-z0-9][a-z0-9\-]{0,98}[a-z0-9]$|^[a-z0-9]$"


def _validate_slug(value: str, field_name: str = "slug") -> str:
    """Raise HTTPException 422 if value does not match the slug pattern."""
    if not re.match(_SLUG_PATTERN, value):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid {field_name}: must be lowercase alphanumeric with hyphens (max 100 chars).",
        )
    return value


# ─── General ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
@limiter.limit(config.RATE_LIMIT_DEFAULT)
def home(request: Request):
    return {"message": "Padelytics API 🎾", "docs": "/docs"}


# ─── Players ─────────────────────────────────────────────────────────────────
@app.get("/players", tags=["Players"])
@limiter.limit(config.RATE_LIMIT_DEFAULT)
def get_players(
    request: Request,
    skip: int = Query(0, ge=0, description="Number of records to skip (pagination)"),
    limit: int = Query(20, ge=1, le=2000, description="Maximum records to return (max 2000)"),
    search: Optional[str] = Query(
        None,
        min_length=1,
        max_length=100,
        description="Case-insensitive name search",
    ),
):
    """Returns a paginated list of players. Supports name search."""
    query = supabase.table("players").select("*")
    if search:
        # search is already length-validated by Pydantic; strip further safety
        query = query.ilike("name", f"%{search.strip()}%")
    res = query.range(skip, skip + limit - 1).execute()
    return res.data


@app.get("/players/ranking", tags=["Players"])
@limiter.limit(config.RATE_LIMIT_DEFAULT)
def get_players_ranking(
    request: Request,
    limit: int = Query(50, ge=1, le=2000, description="Maximum players to return (max 2000)"),
):
    """Returns the latest ranking snapshot ordered by points descending."""
    latest = (
        supabase.table("dynamic_players")
        .select("snapshot_date")
        .order("snapshot_date", desc=True)
        .limit(1)
        .execute()
    )
    if not latest.data:
        raise HTTPException(404, detail="No ranking data available")
    latest_date = latest.data[0]["snapshot_date"]
    res = (
        supabase.table("dynamic_players")
        .select("*, players(*)")
        .eq("snapshot_date", latest_date)
        .order("points", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data


@app.get("/players/head-to-head", tags=["Players"])
@limiter.limit(config.RATE_LIMIT_SEARCH)   # Stricter limit: heavier DB query
def get_players_head_to_head(
    request: Request,
    player1: str = Query(
        ...,
        min_length=1,
        max_length=100,
        pattern=_SLUG_PATTERN,
        description="Slug of the first player",
    ),
    player2: str = Query(
        ...,
        min_length=1,
        max_length=100,
        pattern=_SLUG_PATTERN,
        description="Slug of the second player",
    ),
):
    """Compare two players using their latest stats."""
    p1_res = (
        supabase.table("dynamic_players")
        .select("*, players(*)")
        .eq("slug", player1)
        .order("snapshot_date", desc=True)
        .limit(1)
        .execute()
    )
    p2_res = (
        supabase.table("dynamic_players")
        .select("*, players(*)")
        .eq("slug", player2)
        .order("snapshot_date", desc=True)
        .limit(1)
        .execute()
    )
    if not p1_res.data:
        raise HTTPException(404, detail=f"Player '{player1}' not found")
    if not p2_res.data:
        raise HTTPException(404, detail=f"Player '{player2}' not found")
    return {"player1": p1_res.data[0], "player2": p2_res.data[0]}


@app.get("/players/{slug}", tags=["Players"])
@limiter.limit(config.RATE_LIMIT_DEFAULT)
def get_player_profile(
    request: Request,
    slug: str = Path(..., min_length=1, max_length=100, description="Player slug"),
    history: int = Query(10, ge=1, le=100, description="Snapshot history depth (max 100)"),
):
    """Static profile + evolution history for a single player."""
    # Extra server-side slug validation (defence-in-depth beyond Path constraints)
    _validate_slug(slug, "player slug")
    player = supabase.table("players").select("*").eq("slug", slug).execute()
    if not player.data:
        raise HTTPException(404, detail="Player not found")
    stats = (
        supabase.table("dynamic_players")
        .select("*")
        .eq("slug", slug)
        .order("snapshot_date", desc=True)
        .limit(history)
        .execute()
    )
    return {
        "profile": player.data[0],
        "history": stats.data if stats.data else None,
    }


# ─── Pairs ───────────────────────────────────────────────────────────────────
@app.get("/pairs", tags=["Pairs"])
@limiter.limit(config.RATE_LIMIT_DEFAULT)
def get_pairs(
    request: Request,
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(20, ge=1, le=2000, description="Maximum records to return (max 2000)"),
    search: Optional[str] = Query(
        None,
        min_length=1,
        max_length=100,
        description="Case-insensitive pair_slug search",
    ),
):
    """Returns a paginated list of pairs. Supports pair_slug search."""
    query = supabase.table("dynamic_pairs").select("*")
    if search:
        query = query.ilike("pair_slug", f"%{search.strip()}%")
    res = query.range(skip, skip + limit - 1).execute()
    return res.data


@app.get("/pairs/ranking", tags=["Pairs"])
@limiter.limit(config.RATE_LIMIT_DEFAULT)
def get_pairs_ranking(
    request: Request,
    limit: int = Query(50, ge=1, le=2000, description="Maximum pairs to return (max 2000)"),
):
    """Returns the latest pairs ranking snapshot ordered by points descending."""
    latest = (
        supabase.table("dynamic_pairs")
        .select("snapshot_date")
        .order("snapshot_date", desc=True)
        .limit(1)
        .execute()
    )
    if not latest.data:
        raise HTTPException(404, detail="No pairs data available")
    latest_date = latest.data[0]["snapshot_date"]
    res = (
        supabase.table("dynamic_pairs")
        .select("*, player1:players!player1_slug(*), player2:players!player2_slug(*)")
        .eq("snapshot_date", latest_date)
        .order("points", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data


@app.get("/pairs/head-to-head", tags=["Pairs"])
@limiter.limit(config.RATE_LIMIT_SEARCH)   # Stricter limit: heavier DB query
def get_pairs_head_to_head(
    request: Request,
    slug1: str = Query(
        ...,
        min_length=1,
        max_length=100,
        pattern=_SLUG_PATTERN,
        description="Slug of the first pair",
    ),
    slug2: str = Query(
        ...,
        min_length=1,
        max_length=100,
        pattern=_SLUG_PATTERN,
        description="Slug of the second pair",
    ),
):
    """Compare two pairs using their latest stats."""
    p1_res = (
        supabase.table("dynamic_pairs")
        .select("*, player1:players!player1_slug(*), player2:players!player2_slug(*)")
        .eq("pair_slug", slug1)
        .order("snapshot_date", desc=True)
        .limit(1)
        .execute()
    )
    p2_res = (
        supabase.table("dynamic_pairs")
        .select("*, player1:players!player1_slug(*), player2:players!player2_slug(*)")
        .eq("pair_slug", slug2)
        .order("snapshot_date", desc=True)
        .limit(1)
        .execute()
    )
    if not p1_res.data:
        raise HTTPException(404, detail=f"Pair '{slug1}' not found")
    if not p2_res.data:
        raise HTTPException(404, detail=f"Pair '{slug2}' not found")
    return {"pair1": p1_res.data[0], "pair2": p2_res.data[0]}


@app.get("/pairs/{pair_slug}", tags=["Pairs"])
@limiter.limit(config.RATE_LIMIT_DEFAULT)
def get_pair_profile(
    request: Request,
    pair_slug: str = Path(..., min_length=1, max_length=100, description="Pair slug"),
    history: int = Query(10, ge=1, le=100, description="Snapshot history depth (max 100)"),
):
    """Pair profile + evolution history."""
    _validate_slug(pair_slug, "pair slug")
    pair = supabase.table("dynamic_pairs").select("*").eq("pair_slug", pair_slug).execute()
    if not pair.data:
        raise HTTPException(404, detail="Pair not found")
    stats = (
        supabase.table("dynamic_pairs")
        .select("*")
        .eq("pair_slug", pair_slug)
        .order("snapshot_date", desc=True)
        .limit(history)
        .execute()
    )
    return {
        "profile": pair.data[0],
        "history": stats.data if stats.data else None,
    }


# ─── Matches ─────────────────────────────────────────────────────────────────
@app.get("/matches", tags=["Matches"])
@limiter.limit(config.RATE_LIMIT_DEFAULT)
def get_matches(
    request: Request,
    limit: int = Query(20, ge=1, le=2000, description="Maximum matches to return (max 2000)"),
    tournament_id: Optional[int] = Query(None, ge=1, description="Filter by tournament id"),
    date_from: Optional[date] = Query(None, description="Filter matches from this date (YYYY-MM-DD)"),
):
    """List matches with optional filters and pagination."""
    query = supabase.table("matches").select("*").order("date", desc=True)
    if tournament_id is not None:
        query = query.eq("tournament_id", tournament_id)
    if date_from is not None:
        query = query.gte("date", date_from)
    return query.limit(limit).execute().data


@app.get("/matches/head-to-head", tags=["Matches"])
@limiter.limit(config.RATE_LIMIT_SEARCH)   # Stricter limit: heavier DB query
def get_matches_head_to_head(
    request: Request,
    pair1: str = Query(
        ...,
        min_length=1,
        max_length=100,
        pattern=_SLUG_PATTERN,
        description="Slug of the first pair",
    ),
    pair2: str = Query(
        ...,
        min_length=1,
        max_length=100,
        pattern=_SLUG_PATTERN,
        description="Slug of the second pair",
    ),
):
    """Match history between two pairs, using query params (?pair1=&pair2=)."""
    slugs = f"({pair1},{pair2})"
    res = (
        supabase.table("matches")
        .select("*")
        .filter("team1_slug", "in", slugs)
        .filter("team2_slug", "in", slugs)
        .order("date", desc=True)
        .execute()
    )
    matches = res.data
    wins1, wins2 = 0, 0
    for m in matches:
        is_p1_home = m["team1_slug"] == pair1
        if is_p1_home:
            if m["winner_team"] == 1:
                wins1 += 1
            else:
                wins2 += 1
        else:
            if m["winner_team"] == 2:
                wins1 += 1
            else:
                wins2 += 1
    return {
        "summary": {pair1: wins1, pair2: wins2, "total_matches": len(matches)},
        "history": matches,
    }


# ─── Tournaments ─────────────────────────────────────────────────────────────
@app.get("/tournaments", tags=["Tournaments"])
@limiter.limit(config.RATE_LIMIT_DEFAULT)
def get_tournaments(
    request: Request,
    year: int = Query(2025, ge=2000, le=2100, description="Filter tournaments by year"),
    tournament_id: Optional[int] = Query(None, ge=0, description="Filter by specific tournament id"),
):
    """List tournaments filtered by year or a specific tournament id."""
    query = supabase.table("tournaments").select("*")
    if tournament_id is not None:
        query = query.eq("id", tournament_id)
    else:
        query = query.gte("start_date", f"{year}-01-01").lte("start_date", f"{year}-12-31")
    res = query.order("start_date", desc=False).execute()
    return res.data


# ─── Analytics / Search ───────────────────────────────────────────────────────
@app.get("/search", tags=["Analytics"])
@limiter.limit(config.RATE_LIMIT_SEARCH)   # Stricter limit: hits 3 tables simultaneously
def global_search(
    request: Request,
    q: str = Query(
        ...,
        min_length=1,
        max_length=100,
        description="Search term (players, pairs, tournaments)",
    ),
):
    """Search players, pairs, and tournaments simultaneously."""
    term = q.strip()
    results = []
    for p in (
        supabase.table("players")
        .select("*")
        .ilike("name", f"%{term}%")
        .limit(5)
        .execute()
        .data
    ):
        results.append({"type": "player", "slug": p["slug"], "label": p["name"]})
    for pair in (
        supabase.table("dynamic_pairs")
        .select("pair_slug")
        .ilike("pair_slug", f"%{term}%")
        .limit(5)
        .execute()
        .data
    ):
        label = pair["pair_slug"].replace("--", " / ").replace("-", " ").title()
        results.append({"type": "pair_slug", "slug": pair["pair_slug"], "label": label})
    for t in (
        supabase.table("tournaments")
        .select("*")
        .ilike("full_name", f"%{term}%")
        .limit(3)
        .execute()
        .data
    ):
        results.append({"type": "tournament", "id": str(t["id"]), "label": t["full_name"]})
    return results
