import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx")

app = FastAPI(title="Movie Recommender API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

# ---------------- GLOBALS ----------------
df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None
TITLE_TO_IDX: Optional[Dict[str, int]] = None

# ---------------- MODELS ----------------
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: Optional[List[dict]] = None


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: Optional[TMDBMovieDetails]
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


# ---------------- HELPERS ----------------
def norm_title(t: str) -> str:
    return t.strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG}{path}"


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB request error: {type(e).__name__} | {repr(e)}",
        )

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"TMDB error: {r.text}")

    return r.json()


async def tmdb_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []

    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


# ---------------- INDEX ----------------
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[norm_title(k)] = int(v)
        return title_to_idx

    for k, v in indices.items():
        title_to_idx[norm_title(k)] = int(v)

    return title_to_idx


def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX

    if TITLE_TO_IDX is None:
        raise HTTPException(500, "TF-IDF index map not initialized")

    key = norm_title(title)

    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])

    raise HTTPException(404, f"Title not found: {title}")


def tfidf_recommend_titles(query_title: str, top_n: int = 10) -> List[Tuple[str, float]]:
    global df, tfidf_matrix

    if df is None or tfidf_matrix is None:
        raise HTTPException(500, "TF-IDF resources not loaded")

    idx = get_local_idx_by_title(query_title)

    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()

    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []

    for i in order:
        if int(i) == int(idx):
            continue

        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue

        out.append((title_i, float(scores[int(i)])))

        if len(out) >= top_n:
            break

    return out


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    try:
        m = await tmdb_search_first(title)

        if not m:
            return None

        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )

    except Exception:
        return None


# ---------------- STARTUP ----------------
@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    df = pickle.load(open(DF_PATH, "rb"))
    indices_obj = pickle.load(open(INDICES_PATH, "rb"))
    tfidf_matrix = pickle.load(open(TFIDF_MATRIX_PATH, "rb"))
    tfidf_obj = pickle.load(open(TFIDF_PATH, "rb"))

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    if df is None or "title" not in df.columns:
        raise RuntimeError("df must contain title column")


# ---------------- ROUTES ----------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/home", response_model=List[TMDBMovieCard])
async def home(category: str = "popular", limit: int = 24):
    data = await tmdb_get(f"/movie/{category}", {"language": "en-US"})
    return await tmdb_cards_from_results(data.get("results", []), limit)


@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)


@app.get("/tmdb/search")
async def tmdb_search(query: str):
    return await tmdb_search_movies(query)


@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(tmdb_id: int, limit: int = 18):
    details = await tmdb_movie_details(tmdb_id)

    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]

    data = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )

    return await tmdb_cards_from_results(data.get("results", []), limit)


@app.get("/recommend/tfidf")
async def recommend_tfidf(title: str, top_n: int = 10):
    recs = tfidf_recommend_titles(title, top_n)

    return [{"title": t, "score": s} for t, s in recs]


@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(query: str):
    best = await tmdb_search_first(query)

    if not best:
        raise HTTPException(404, "Not found")

    details = await tmdb_movie_details(best["id"])

    tfidf_items = []
    recs = tfidf_recommend_titles(details.title, 10)

    for t, s in recs:
        card = await attach_tmdb_card_by_title(t)
        tfidf_items.append(TFIDFRecItem(title=t, score=s, tmdb=card))

    genre_recs = []

    if details.genres:
        genre_id = details.genres[0]["id"]

        data = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )

        genre_recs = await tmdb_cards_from_results(
            data.get("results", []), 12
        )

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )
print("TMDB KEY:", TMDB_API_KEY)
