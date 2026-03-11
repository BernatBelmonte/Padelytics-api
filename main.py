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

import io
import math
import re
import warnings
from contextlib import asynccontextmanager
from datetime import date
from typing import List, Optional

import httpx
import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from supabase import Client, ClientOptions, create_client

import config

#
# Custom httpx.Client to prevent httpx.ReadError (EAGAIN) on Render:
#   keepalive_expiry=20  – evicts idle connections after 20 s, well below Render's
#                          ~55 s idle-close threshold, so we never reuse a stale socket.
#   HTTPTransport(retries=1) – auto-retries once on any remaining transient error.
#   Explicit timeouts     – avoids silent hangs on slow cold starts.
_http_client = httpx.Client(
    transport=httpx.HTTPTransport(retries=1),
    timeout=httpx.Timeout(30.0, connect=10.0),
    limits=httpx.Limits(
        max_connections=10,
        max_keepalive_connections=5,
        keepalive_expiry=20.0,
    ),
)

supabase: Client = create_client(
    config.SUPABASE_URL,
    config.SUPABASE_KEY,
    options=ClientOptions(httpx_client=_http_client),
)

# ─── Rate limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[config.RATE_LIMIT_DEFAULT],
)

_docs_url = "/docs" if config.APP_ENV != "production" else None
_redoc_url = "/redoc" if config.APP_ENV != "production" else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cleanly close the shared httpx.Client on shutdown so OS sockets are released.
    _http_client.close()


app = FastAPI(
    title="Padelytics API – Padel Pro Analytics",
    description="Advanced API for accessing professional padel data from Premier Padel.",
    version="2.0.0",
    docs_url=_docs_url,
    redoc_url=_redoc_url,
    lifespan=lifespan,
)

# ─── Rate-limit exception handler ────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ─── CORS middleware ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=False,          # No cookies/sessions used
    allow_methods=["GET", "POST"],    # POST required for /simulate
    allow_headers=["Content-Type", "Authorization"],
)


# ─── Security headers middleware ──────────────────────────────────────────────
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
    """
    API health-check and entry point.

    Returns a welcome message and the URL to the interactive documentation.
    Use this endpoint to verify the service is running.
    """
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
    """
    Return a paginated list of all players in the database.

    **Pagination**: use `skip` and `limit` to page through results.

    **Search**: supply `search` for a case-insensitive partial match on the player's name.

    **Response**: array of player objects from the `players` table.
    """
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
    """
    Return the latest official player ranking snapshot.

    Looks up the most recent `snapshot_date` in `dynamic_players` and returns
    players ordered by `points` descending. Each row embeds the static player
    profile from the `players` table.

    **Response**: array of `dynamic_players` rows with a nested `players` object.
    """
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
    """
    Compare two players side-by-side using their most recent stats snapshot.

    Each player is identified by their URL `slug`. The endpoint fetches the
    latest `dynamic_players` row for each slug, embedding the static player
    profile.

    **Query parameters**:
    - `player1`: slug of the first player (required)
    - `player2`: slug of the second player (required)

    **Response**: `{ player1: {...}, player2: {...} }` — each value is the latest
    `dynamic_players` row with a nested `players` object.
    """
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
    """
    Return the static profile and ranking evolution history for a single player.

    The `profile` section comes from the `players` table (immutable bio data).
    The `history` section is up to `history` snapshots from `dynamic_players`,
    ordered most-recent first, allowing you to chart points or rank over time.

    **Path parameter**:
    - `slug`: player slug (lowercase alphanumeric with hyphens)

    **Query parameter**:
    - `history`: number of past snapshots to return (1–100, default 10)

    **Response**: `{ profile: {...}, history: [...] }`
    """
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
    """
    Return a paginated list of pairs from the `dynamic_pairs` table.

    **Pagination**: use `skip` and `limit` to page through results.

    **Search**: supply `search` for a case-insensitive partial match on `pair_slug`.

    **Response**: array of `dynamic_pairs` rows (latest snapshot per pair).
    """
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
    """
    Return the latest official pairs ranking snapshot.

    Looks up the most recent `snapshot_date` in `dynamic_pairs` and returns
    pairs ordered by `points` descending. Each row embeds both player profiles
    from the `players` table via the `player1_slug` and `player2_slug` foreign keys.

    **Response**: array of `dynamic_pairs` rows with nested `player1` and `player2` objects.
    """
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
    """
    Compare two pairs side-by-side using their most recent stats snapshot.

    Each pair is identified by its `pair_slug`. The endpoint fetches the latest
    `dynamic_pairs` row for each slug, embedding both player profiles.

    Note: stats here are **pre-aggregated across all matches** with no context
    filters. For environment- or round-filtered analysis use `/pairs/contextual-stats`.

    **Query parameters**:
    - `slug1`: slug of the first pair (required)
    - `slug2`: slug of the second pair (required)

    **Response**: `{ pair1: {...}, pair2: {...} }` — each value is the latest
    `dynamic_pairs` row with nested `player1` and `player2` objects.
    """
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


# ─── Contextual stats helper ──────────────────────────────────────────────────

def _pair_wins_losses(matches: list, pair_slug: str) -> dict:
    """Compute win/loss/total/win_rate for one pair from a list of match rows."""
    wins = losses = 0
    for m in matches:
        if m["team1_slug"] == pair_slug:
            if m["winner_team"] == 1:
                wins += 1
            else:
                losses += 1
        elif m["team2_slug"] == pair_slug:
            if m["winner_team"] == 2:
                wins += 1
            else:
                losses += 1
    total = wins + losses
    return {
        "wins": wins,
        "losses": losses,
        "total_matches": total,
        "win_rate": round(wins / total, 3) if total else None,
    }


@app.get("/pairs/contextual-stats", tags=["Pairs"])
@limiter.limit(config.RATE_LIMIT_SEARCH)
def get_pairs_contextual_stats(
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
    tournament_level: Optional[List[str]] = Query(
        default=None,
        description="List of tournament levels: FINALS, MAJOR, P1, or P2",
    ),
    venue_type: Optional[List[str]] = Query(
        default=None,
        description="List of venue types: indoor or outdoor",
    ),
    altitude_min: Optional[int] = Query(None, ge=0, description="Minimum venue altitude (metres)"),
    altitude_max: Optional[int] = Query(None, ge=0, description="Maximum venue altitude (metres)"),
    temp_min: Optional[float] = Query(None, description="Minimum average temperature (°C)"),
    temp_max: Optional[float] = Query(None, description="Maximum average temperature (°C)"),
    humidity_min: Optional[float] = Query(None, ge=0, le=100, description="Minimum average humidity (%)"),
    humidity_max: Optional[float] = Query(None, ge=0, le=100, description="Maximum average humidity (%)"),
    court_speed_min: Optional[float] = Query(None, ge=0, description="Minimum court speed index"),
    court_speed_max: Optional[float] = Query(None, ge=0, description="Maximum court speed index"),
    round_name: Optional[List[str]] = Query(
        default=None,
        description="List of round stages: Men F, Men SF, Men QF, Men R16, or Men R32",
    ),
):
    """
    Compare two pairs within a specific environmental and/or round context.

    Stats are recomputed from raw match data — not from pre-aggregated snapshots —
    so every active filter is applied before wins/losses are counted.

    **Tournament filters** (applied via the tournaments table):
    tournament_level, venue_type, altitude, avg_temperature, avg_humidity, court_speed_index

    **Match filter**: round_name

    **Response sections**:
    - `head_to_head`: record when the two pairs faced *each other* inside the context
    - `individual_context`: each pair's overall record vs *any* opponent inside the context
    - `matches_in_context`: raw head-to-head match list used for the H2H calculation
    """
    _VALID_LEVELS = {"FINALS", "MAJOR", "P1", "P2"}
    _VALID_VENUES = {"indoor", "outdoor"}
    _VALID_ROUNDS = {"Men F", "Men SF", "Men QF", "Men R16", "Men R32"}
    if tournament_level:
        invalid = [v for v in tournament_level if v not in _VALID_LEVELS]
        if invalid:
            raise HTTPException(status_code=422, detail=f"Invalid tournament_level value(s): {invalid}")
    if venue_type:
        invalid = [v for v in venue_type if v not in _VALID_VENUES]
        if invalid:
            raise HTTPException(status_code=422, detail=f"Invalid venue_type value(s): {invalid}")
    if round_name:
        invalid = [v for v in round_name if v not in _VALID_ROUNDS]
        if invalid:
            raise HTTPException(status_code=422, detail=f"Invalid round_name value(s): {invalid}")

    t_query = supabase.table("tournaments").select("id")
    if tournament_level:
        t_query = t_query.in_("tournament_level", tournament_level)
    if venue_type:
        t_query = t_query.in_("venue_type", venue_type)
    if altitude_min is not None:
        t_query = t_query.gte("altitude", altitude_min)
    if altitude_max is not None:
        t_query = t_query.lte("altitude", altitude_max)
    if temp_min is not None:
        t_query = t_query.gte("avg_temperature", temp_min)
    if temp_max is not None:
        t_query = t_query.lte("avg_temperature", temp_max)
    if humidity_min is not None:
        t_query = t_query.gte("avg_humidity", humidity_min)
    if humidity_max is not None:
        t_query = t_query.lte("avg_humidity", humidity_max)
    if court_speed_min is not None:
        t_query = t_query.gte("court_speed_index", court_speed_min)
    if court_speed_max is not None:
        t_query = t_query.lte("court_speed_index", court_speed_max)

    t_res = t_query.execute()
    if not t_res.data:
        return {
            "filters_matched_tournaments": 0,
            "message": "No tournaments match the given environmental filters.",
            "applied_filters": {
                "tournament_level": tournament_level,
                "venue_type": venue_type,
                "altitude_min": altitude_min,
                "altitude_max": altitude_max,
                "temp_min": temp_min,
                "temp_max": temp_max,
                "humidity_min": humidity_min,
                "humidity_max": humidity_max,
                "court_speed_min": court_speed_min,
                "court_speed_max": court_speed_max,
                "round_name": round_name,
            },
            "head_to_head": None,
            "individual_context": {slug1: None, slug2: None},
            "matches_in_context": [],
        }

    tournament_ids_str = "({})".format(",".join(str(t["id"]) for t in t_res.data))

    slugs_str = f"({slug1},{slug2})"
    h2h_query = (
        supabase.table("matches")
        .select("*")
        .filter("team1_slug", "in", slugs_str)
        .filter("team2_slug", "in", slugs_str)
        .filter("tournament_id", "in", tournament_ids_str)
        .order("date", desc=True)
    )
    if round_name:
        h2h_query = h2h_query.in_("round_name", round_name)
    h2h_matches = h2h_query.execute().data

    def _individual_record(slug: str) -> dict:
        q = (
            supabase.table("matches")
            .select("winner_team, team1_slug, team2_slug")
            .or_(f"team1_slug.eq.{slug},team2_slug.eq.{slug}")
            .filter("tournament_id", "in", tournament_ids_str)
        )
        if round_name:
            q = q.in_("round_name", round_name)
        return _pair_wins_losses(q.execute().data, slug)

    slug1_h2h = _pair_wins_losses(h2h_matches, slug1)
    slug2_h2h = _pair_wins_losses(h2h_matches, slug2)

    return {
        "filters_matched_tournaments": len(t_res.data),
        "applied_filters": {
            "tournament_level": tournament_level,
            "venue_type": venue_type,
            "altitude_min": altitude_min,
            "altitude_max": altitude_max,
            "temp_min": temp_min,
            "temp_max": temp_max,
            "humidity_min": humidity_min,
            "humidity_max": humidity_max,
            "court_speed_min": court_speed_min,
            "court_speed_max": court_speed_max,
            "round_name": round_name,
        },
        "head_to_head": {
            "summary": {
                slug1: slug1_h2h["wins"],
                slug2: slug2_h2h["wins"],
                "total_matches": len(h2h_matches),
            },
            "detail": {
                slug1: slug1_h2h,
                slug2: slug2_h2h,
            },
        },
        "individual_context": {
            slug1: _individual_record(slug1),
            slug2: _individual_record(slug2),
        },
        "matches_in_context": h2h_matches,
    }


@app.get("/pairs/{pair_slug}", tags=["Pairs"])
@limiter.limit(config.RATE_LIMIT_DEFAULT)
def get_pair_profile(
    request: Request,
    pair_slug: str = Path(..., min_length=1, max_length=100, description="Pair slug"),
    history: int = Query(10, ge=1, le=100, description="Snapshot history depth (max 100)"),
):
    """
    Return the profile and ranking evolution history for a single pair.

    The `profile` section is the most recent `dynamic_pairs` row for the pair.
    The `history` section contains up to `history` snapshots ordered most-recent
    first, allowing you to chart points, win rate, or rank over time.

    **Path parameter**:
    - `pair_slug`: pair slug (lowercase alphanumeric with hyphens)

    **Query parameter**:
    - `history`: number of past snapshots to return (1–100, default 10)

    **Response**: `{ profile: {...}, history: [...] }`
    """
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
    """
    Return a list of matches ordered by date descending.

    **Filters**:
    - `tournament_id`: restrict to a single tournament
    - `date_from`: only return matches on or after this date (YYYY-MM-DD)

    **Pagination**: use `limit` to cap the number of results (max 2000).

    **Response**: array of match rows from the `matches` table.
    """
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
    """
    Return the full head-to-head match history between two pairs.

    Fetches every match where both `pair1` and `pair2` appeared (in either
    `team1_slug` or `team2_slug`), then computes win totals for each side.

    **Query parameters**:
    - `pair1`: slug of the first pair (required)
    - `pair2`: slug of the second pair (required)

    **Response**:
    - `summary`: `{ pair1_slug: wins, pair2_slug: wins, total_matches: N }`
    - `history`: full list of match rows ordered by date descending
    """
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
    """
    Return a list of tournaments ordered by start date ascending.

    **Filters** (mutually exclusive — `tournament_id` takes priority):
    - `tournament_id`: return a single specific tournament by its id
    - `year`: return all tournaments whose `start_date` falls within that year
      (default: 2025)

    **Response**: array of tournament rows including venue metadata
    (altitude, avg_temperature, avg_humidity, court_speed_index, venue_type, etc.)
    """
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
    """
    Search players, pairs, and tournaments simultaneously with a single query.

    The search term is matched case-insensitively against:
    - `players.name` → returns up to 5 player results
    - `dynamic_pairs.pair_slug` → returns up to 5 pair results
    - `tournaments.full_name` → returns up to 3 tournament results

    **Query parameter**:
    - `q`: search term (1–100 characters, required)

    **Response**: flat array of result objects, each with a `type` field
    (`"player"`, `"pair_slug"`, or `"tournament"`), a `slug` or `id`, and a
    human-readable `label`.
    """
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


# ─── AI / Match Simulation ────────────────────────────────────────────────────
_model_cache = None

def _get_model():
    """Download and cache the padel prediction model from Supabase Storage."""
    global _model_cache
    if _model_cache is None:
        model_bytes = supabase.storage.from_("models").download("final_padel_model.pkl")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
            _model_cache = joblib.load(io.BytesIO(model_bytes))
    return _model_cache


def _calculate_smart_speed_index(
    venue_type: str,
    altitude: float,
    avg_temperature: float,
    avg_humidity: float,
) -> float:
    """Calculates a Smart Court Speed Index based on venue and environmental factors."""
    is_indoor = "indoor" in venue_type.lower()

    altitude_score = altitude * 0.012

    temp = avg_temperature
    if temp < 10:
        raw_temp_score = (temp - 10) * 1.5
    elif temp <= 22:
        raw_temp_score = (temp - 10) * 0.5
    else:
        raw_temp_score = 6 + (temp - 22) * 2.0

    raw_hum_penalty = (avg_humidity - 45) * 0.5 if avg_humidity > 45 else 0.0

    weather_weight = 0.2 if is_indoor else 1.0
    final_temp_score = raw_temp_score * weather_weight
    final_hum_penalty = raw_hum_penalty * weather_weight

    indoor_bonus = 5.0 if is_indoor else 0.0

    return 50.0 + altitude_score + final_temp_score - final_hum_penalty + indoor_bonus

class SimulateRequest(BaseModel):
    pair1_slug: str = Field(..., min_length=1, max_length=100, pattern=_SLUG_PATTERN)
    pair2_slug: str = Field(..., min_length=1, max_length=100, pattern=_SLUG_PATTERN)
    venue_type: str = Field(..., pattern=r"^(indoor|outdoor)$")
    altitude: float = Field(..., ge=0, le=5000)
    avg_temperature: float = Field(..., ge=-20, le=60)
    avg_humidity: float = Field(..., ge=0, le=100)


@app.post("/simulate", tags=["AI"])
@limiter.limit(config.RATE_LIMIT_SEARCH)
def simulate_match(request: Request, body: SimulateRequest):
    """
    Predict the winner of a match between two pairs under specific venue conditions.

    The endpoint fetches the latest stats for each pair, computes all model
    features server-side (including court speed index and height diff), loads
    the pre-trained model from Supabase Storage, and returns win probabilities.

    **Body parameters**:
    - `pair1_slug` / `pair2_slug`: pair slugs (required)
    - `venue_type`: `"indoor"` or `"outdoor"` (required)
    - `altitude`: metres above sea level (0–5000, required)
    - `avg_temperature`: °C (-20–60, required)
    - `avg_humidity`: % (0–100, required)

    **Response**: win probabilities and predicted winner from pair1's perspective.
    """
    p1_res = (
        supabase.table("dynamic_pairs")
        .select("*")
        .eq("pair_slug", body.pair1_slug)
        .order("snapshot_date", desc=True)
        .limit(1)
        .execute()
    )
    p2_res = (
        supabase.table("dynamic_pairs")
        .select("*")
        .eq("pair_slug", body.pair2_slug)
        .order("snapshot_date", desc=True)
        .limit(1)
        .execute()
    )
    if not p1_res.data:
        raise HTTPException(404, detail=f"Pair '{body.pair1_slug}' not found")
    if not p2_res.data:
        raise HTTPException(404, detail=f"Pair '{body.pair2_slug}' not found")

    p1 = p1_res.data[0]
    p2 = p2_res.data[0]

    # 2. Fetch player heights for all four players
    all_slugs = list({
        p1["player1_slug"], p1["player2_slug"],
        p2["player1_slug"], p2["player2_slug"],
    })
    players_res = (
        supabase.table("players")
        .select("slug, height")
        .in_("slug", all_slugs)
        .execute()
    )
    height_map: dict = {row["slug"]: row.get("height") for row in players_res.data}

    h_p1_a = height_map.get(p1["player1_slug"])
    h_p1_b = height_map.get(p1["player2_slug"])
    h_p2_a = height_map.get(p2["player1_slug"])
    h_p2_b = height_map.get(p2["player2_slug"])

    # 3. Court speed index
    court_speed = _calculate_smart_speed_index(
        body.venue_type, body.altitude, body.avg_temperature, body.avg_humidity
    )

    # 4. Build feature vector
    def _f(val):
        """Return float, or NaN if None."""
        return float("nan") if val is None else float(val)

    def _f0(val):
        """Return float, or 0.0 if None (safe default for optional rate stats)."""
        return 0.0 if val is None else float(val)

    p1_pts = _f(p1.get("points"))
    p2_pts = _f(p2.get("points"))

    log_diff = (
        math.log(p1_pts) - math.log(p2_pts)
        if p1_pts > 0 and p2_pts > 0
        else float("nan")
    )

    if None in (h_p1_a, h_p1_b, h_p2_a, h_p2_b):
        diff_avg_height = 0.0
    else:
        diff_avg_height = ((h_p1_a + h_p1_b) / 2) - ((h_p2_a + h_p2_b) / 2)

    features = [
        p1_pts + p2_pts,
        court_speed,
        log_diff,
        _f(p1.get("points_change")) - _f(p2.get("points_change")),
        _f(p1.get("tournaments_played_together")) - _f(p2.get("tournaments_played_together")),
        _f(p1.get("matches_last_14_days")) - _f(p2.get("matches_last_14_days")),
        _f0(p1.get("finals_conversion_rate")) - _f0(p2.get("finals_conversion_rate")),
        _f0(p1.get("win_pct")) - _f0(p2.get("win_pct")),
        _f0(p1.get("avg_games_conceded_per_set")) - _f0(p2.get("avg_games_conceded_per_set")),
        _f0(p1.get("tie_break_win_pct")) - _f0(p2.get("tie_break_win_pct")),
        _f0(p1.get("comeback_rate")) - _f0(p2.get("comeback_rate")),
        diff_avg_height,
    ]

    if any(math.isnan(f) for f in features):
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Simulation could not be completed due to insufficient data. "
                    "One or more required statistics are missing for the selected pairs."
                ),
            },
        )

    _MODEL_COLUMNS = [
        "match_quality_sum",
        "court_speed_index",
        "diff_log_total_points",
        "diff_points_change",
        "diff_tournaments_played_together",
        "diff_matches_last_14_days",
        "diff_finals_conversion_rate",
        "diff_season_win_pct",
        "diff_avg_games_conceded_per_set",
        "diff_tie_break_win_pct",
        "diff_comeback_rate",
        "diff_avg_height",
    ]
    model = _get_model()
    X = pd.DataFrame([features], columns=_MODEL_COLUMNS)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        # Class 1 = pair1 wins, class 0 = pair2 wins
        pair1_win_prob = round(float(proba[1]), 4)
        pair2_win_prob = round(float(proba[0]), 4)
    else:
        pred = int(model.predict(X)[0])
        pair1_win_prob = 1.0 if pred == 1 else 0.0
        pair2_win_prob = 1.0 - pair1_win_prob

    predicted_winner = body.pair1_slug if pair1_win_prob >= 0.5 else body.pair2_slug

    return {
        "pair1_slug": body.pair1_slug,
        "pair2_slug": body.pair2_slug,
        "pair1_win_probability": pair1_win_prob,
        "pair2_win_probability": pair2_win_prob,
        "predicted_winner": predicted_winner,
    }
