import streamlit as st
import requests

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="CineAI", layout="wide")

TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

# ======================
# API FUNCTIONS
# ======================

def fetch_movies(category):
    try:
        if category == "trending":
            url = "https://api.themoviedb.org/3/trending/movie/day"
            params = {"api_key": TMDB_API_KEY, "language": "en-US"}
        else:
            url = f"https://api.themoviedb.org/3/movie/{category}"
            params = {
                "api_key": TMDB_API_KEY,
                "language": "en-US",
                "page": 1
            }

        res = requests.get(url, params=params, timeout=10)
        data = res.json()

        return data.get("results", [])

    except Exception as e:
        st.error(f"API Error: {e}")
        return []


def search_movies(query):
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "query": query,
            "language": "en-US"
        }

        res = requests.get(url, params=params, timeout=10)
        return res.json().get("results", [])

    except Exception as e:
        st.error(f"Search Error: {e}")
        return []


# ======================
# UI HEADER
# ======================
st.title("🎬 CineAI - Movie Recommendation System")
st.caption("Built with ❤️ by Shagun")

# ======================
# SIDEBAR
# ======================
st.sidebar.header("🔥 Explore Movies")

category = st.sidebar.selectbox(
    "Select Category",
    ["popular", "top_rated", "upcoming", "now_playing", "trending"]
)

search_query = st.sidebar.text_input("Search Movies")

# ======================
# DATA FETCH
# ======================
if search_query:
    movies = search_movies(search_query)
else:
    movies = fetch_movies(category)

# ======================
# DEBUG (optional remove later)
# ======================
# st.write(movies)

# ======================
# DISPLAY MOVIES
# ======================
st.subheader(f"Showing results for: {search_query if search_query else category}")

if not movies:
    st.warning("No movies found 😢")
else:
    cols = st.columns(5)

    for i, movie in enumerate(movies):
        with cols[i % 5]:
            poster = movie.get("poster_path")

            if poster:
                st.image(TMDB_IMG + poster)
            else:
                st.image("https://via.placeholder.com/300x450")

            st.write(movie.get("title", "No Title"))
            st.caption(f"⭐ {movie.get('vote_average', 'N/A')}")