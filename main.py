import os
from datetime import date
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")

if not url or not key:
    raise ValueError(" SUPABASE_URL and SUPABASE_KEY must be set in .env file")

supabase: Client = create_client(url, key)

app = FastAPI(
    title="Padelytics API Padel Pro Analytics",
    description="Advanced API for accessing professional padel data from Premier Padel.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── General ────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def home():
    return {"message": "Padelytics API 🎾", "docs": "/docs"}

# ─── Players ────────────────────────────────────────────────────────────────
# /players                                  - list with search
# /players/{slug}                           - profile + evolution
# /players/ranking                          - official ranking (latest snapshot)
# /players/headtohead?player1=&player2=     - compare 2 players (latest snapshot)

@app.get("/players", tags=["Players"])
def get_players(skip: int = 0, limit: int = 20, search: Optional[str] = None):
    """Returns a list of players. Supports name search."""
    query = supabase.table("players").select("*")
    if search:
        query = query.ilike("name", f"%{search}%")
    res = query.range(skip, skip + limit - 1).execute()
    return res.data

@app.get("/players/ranking", tags=["Players"])
def get_players_ranking(limit: int = 50):
    """Returns latest ranking snapshot ordered by points descending."""
    latest = supabase.table("dynamic_players") \
        .select("snapshot_date").order("snapshot_date", desc=True).limit(1).execute()
    if not latest.data:
        raise HTTPException(404, detail="No ranking data available")
    latest_date = latest.data[0]["snapshot_date"]
    res = supabase.table("dynamic_players") \
        .select("*, players(*)") \
        .eq("snapshot_date", latest_date) \
        .order("points", desc=True) \
        .limit(limit) \
        .execute()
    return res.data

@app.get("/players/head-to-head", tags=["Players"])
def get_players_head_to_head(player1: str = Query(...), player2: str = Query(...)):
    """Compare two players using their latest stats. Returns both profiles and the date of the snapshot used for comparison."""
    p1_res = supabase.table("dynamic_players") \
        .select("*, players(*)") \
        .eq("slug", player1) \
        .order("snapshot_date", desc=True) \
        .limit(1).execute()
    p2_res = supabase.table("dynamic_players") \
        .select("*, players(*)") \
        .eq("slug", player2) \
        .order("snapshot_date", desc=True) \
        .limit(1).execute()
    if not p1_res.data:
        raise HTTPException(404, detail=f"Player '{player1}' not found")
    if not p2_res.data:
        raise HTTPException(404, detail=f"Player '{player2}' not found")
    return {"player1": p1_res.data[0], "player2": p2_res.data[0]}


@app.get("/players/{slug}", tags=["Players"])
def get_player_profile(slug: str, history: int = Query(10, description="Include evolution history over time")):
    """Static profile + evolution."""
    player = supabase.table("players").select("*").eq("slug", slug).execute()
    if not player.data:
        raise HTTPException(404, detail="Player not found")
    stats = supabase.table("dynamic_players") \
        .select("*").eq("slug", slug).order("snapshot_date", desc=True).limit(history).execute()
    return {
        "profile": player.data[0],
        "history": stats.data if stats.data else None
    }

# ─── Pairs ──────────────────────────────────────────────────────────────────

@app.get("/pairs", tags=["Pairs"])
def get_pairs(skip: int = 0, limit: int = 20, search: Optional[str] = None):
    """Returns a list of pairs. Supports pair_slug search."""
    query = supabase.table("dynamic_pairs").select("*")
    if search:
        query = query.ilike("pair_slug", f"%{search}%")
    res = query.range(skip, skip + limit - 1).execute()
    return res.data

@app.get("/pairs/ranking", tags=["Pairs"])
def get_pairs_ranking(limit: int = 50):
    """Returns latest ranking snapshot ordered by points descending."""
    latest = supabase.table("dynamic_pairs") \
        .select("snapshot_date").order("snapshot_date", desc=True).limit(1).execute()
    if not latest.data:
        raise HTTPException(404, detail="No pairs data available")
    latest_date = latest.data[0]["snapshot_date"]
    res = supabase.table("dynamic_pairs") \
        .select("*, player1:players!player1_slug(*), player2:players!player2_slug(*)") \
        .eq("snapshot_date", latest_date) \
        .order("points", desc=True) \
        .limit(limit).execute()
    return res.data

@app.get("/pairs/head-to-head", tags=["Pairs"])
def get_pairs_head_to_head(slug1: str = Query(...), slug2: str = Query(...)):
    """
    Compare two pairs using their latest stats. Returns both profiles and the date of the snapshot used for comparison.
    """
    p1_res = supabase.table("dynamic_pairs") \
        .select("*, player1:players!player1_slug(*), player2:players!player2_slug(*)") \
        .eq("pair_slug", slug1) \
        .order("snapshot_date", desc=True).limit(1).execute()
    p2_res = supabase.table("dynamic_pairs") \
        .select("*, player1:players!player1_slug(*), player2:players!player2_slug(*)") \
        .eq("pair_slug", slug2) \
        .order("snapshot_date", desc=True).limit(1).execute()
    if not p1_res.data:
        raise HTTPException(404, detail=f"Pair '{slug1}' not found")
    if not p2_res.data:
        raise HTTPException(404, detail=f"Pair '{slug2}' not found")
    return {
        "pair1": p1_res.data[0],
        "pair2": p2_res.data[0]
    }

@app.get("/pairs/{pair_slug}", tags=["Pairs"])
def get_pair_profile(pair_slug: str, history: int = Query(10, description="Include evolution history over time")):
    """Pair profile + evolution."""
    pair = supabase.table("dynamic_pairs").select("*").eq("pair_slug", pair_slug).execute()
    if not pair.data:
        raise HTTPException(404, detail="Pair not found")
    stats = supabase.table("dynamic_pairs") \
        .select("*").eq("pair_slug", pair_slug).order("snapshot_date", desc=True).limit(history).execute()
    return {
        "profile": pair.data[0],
        "history": stats.data if stats.data else None
    }

# ─── Matches ────────────────────────────────────────────────────────────────

@app.get("/matches", tags=["Matches"])
def get_matches(
    limit: int = Query(20, ge=1, le=200, description="Maximum number of matches to return"),
    tournament_id: Optional[int] = Query(None, description="Filter by tournament id"),
    date_from: Optional[date] = Query(None, description="Filter matches from this date (YYYY-MM-DD)")
):
    """List matches with optional filters and pagination."""
    query = supabase.table("matches").select("*").order("date", desc=True)
    if tournament_id is not None:
        query = query.eq("tournament_id", tournament_id)
    if date_from is not None:
        query = query.gte("date", date_from)
    return query.limit(limit).execute().data


@app.get("/matches/head-to-head", tags=["Matches"])
def get_matches_head_to_head(pair1: str = Query(...), pair2: str = Query(...)):
    """Match history between two pairs/teams, using query params (?pair1=&pair2=)."""
    slugs = f"({pair1},{pair2})"
    res = supabase.table("matches") \
        .select("*") \
        .filter("team1_slug", "in", slugs) \
        .filter("team2_slug", "in", slugs) \
        .order("date", desc=True).execute()
    matches = res.data
    wins1, wins2 = 0, 0
    for m in matches:
        is_p1_home = m["team1_slug"] == pair1
        if is_p1_home:
            if m["winner_team"] == 1: wins1 += 1
            else: wins2 += 1
        else:
            if m["winner_team"] == 2: wins1 += 1
            else: wins2 += 1
    return {"summary": {pair1: wins1, pair2: wins2, "total_matches": len(matches)}, "history": matches}

# ─── Tournaments ────────────────────────────────────────────────────────────

@app.get("/tournaments", tags=["Tournaments"])
def get_tournaments(
    year: int = Query(2025, description="Filter tournaments by year"),
    tournament_id: Optional[int] = Query(None, description="Filter by specific tournament id")
):
    query = supabase.table("tournaments").select("*") \
        .gte("start_date", f"{year}-01-01") \
        .lte("start_date", f"{year}-12-31")
    if tournament_id is not None:
        query = query.eq("id", tournament_id)
    res = query.order("start_date", desc=False).execute()
    return res.data

# ─── Analytics ──────────────────────────────────────────────────────────────

@app.get("/search", tags=["Analytics"])
def global_search(q: str):
    """Search players, pairs, and tournaments simultaneously."""
    results = []
    for p in supabase.table("players").select("*").ilike("name", f"%{q}%").limit(5).execute().data:
        results.append({"type": "player", "slug": p["slug"], "label": p["name"]})
    for pair in supabase.table("dynamic_pairs").select("pair_slug").ilike("pair_slug", f"%{q}%").limit(5).execute().data:
        label = pair["pair_slug"].replace("--", " / ").replace("-", " ").title()
        results.append({"type": "pair_slug", "slug": pair["pair_slug"], "label": label})
    for t in supabase.table("tournaments").select("*").ilike("full_name", f"%{q}%").limit(3).execute().data:
        results.append({"type": "tournament", "id": str(t["id"]), "label": t["full_name"]})
    return results