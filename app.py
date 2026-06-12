import streamlit as st
import requests

BASE_URL = "http://127.0.0.1:8000"
IMG = "https://image.tmdb.org/t/p/w500"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CineAI by Shagun",
    layout="wide"
)

# ---------------- HEADER ----------------
st.markdown("## 🎬 CineAI - Movie Recommendation System")
st.markdown("### Built with ❤️ by Shagun")

# ---------------- SESSION STATE ----------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

# ---------------- API FUNCTIONS ----------------
def fetch_home(category):
    try:
        res = requests.get(f"{BASE_URL}/home", params={"category": category})
        return res.json() if res.status_code == 200 else []
    except:
        return []

def search_movies(query):
    try:
        res = requests.get(f"{BASE_URL}/tmdb/search", params={"query": query})
        return res.json().get("results", [])
    except:
        return []

def get_details(mid):
    try:
        return requests.get(f"{BASE_URL}/movie/id/{mid}").json()
    except:
        return {}

def get_genre(mid):
    try:
        return requests.get(f"{BASE_URL}/recommend/genre", params={"tmdb_id": mid}).json()
    except:
        return []

def get_tfidf(title):
    try:
        return requests.get(f"{BASE_URL}/recommend/tfidf", params={"title": title}).json()
    except:
        return []

# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔍 Search", "🎯 Details", "❤️ Watchlist"]
)

categories = ["popular", "top_rated", "upcoming", "now_playing", "trending"]

# ---------------- HOME ----------------
if menu == "🏠 Home":

    st.subheader("🔥 Explore Movies")

    category = st.selectbox("Select Category", categories)

    movies = fetch_home(category) or []

    cols = st.columns(6)

    for i, m in enumerate(movies):
        with cols[i % 6]:

            if m.get("poster_url"):
                st.image(m["poster_url"], use_container_width=True)
            else:
                st.text("No Image")

            st.caption(m.get("title", "Unknown"))

            if st.button("View", key=f"view_{i}"):
                st.session_state.selected_movie = m.get("tmdb_id")

            if st.button("❤️", key=f"like_{i}"):
                st.session_state.watchlist.append(m)

# ---------------- SEARCH ----------------
elif menu == "🔍 Search":

    st.subheader("Search Movies")

    query = st.text_input("Enter movie name")

    if query:
        results = search_movies(query)

        cols = st.columns(6)

        for i, m in enumerate(results):
            with cols[i % 6]:

                if m.get("poster_path"):
                    st.image(f"{IMG}{m['poster_path']}", use_container_width=True)

                st.caption(m.get("title"))

                if st.button("Open", key=f"search_{i}"):
                    st.session_state.selected_movie = m.get("id")

# ---------------- DETAILS ----------------
elif menu == "🎯 Details":

    if not st.session_state.selected_movie:
        st.warning("Please select a movie first from Home or Search.")
    else:

        mid = st.session_state.selected_movie
        details = get_details(mid)

        col1, col2 = st.columns([1, 2])

        with col1:
            if details.get("poster_url"):
                st.image(details["poster_url"], use_container_width=True)

        with col2:
            st.title(details.get("title"))
            st.write(details.get("overview"))
            st.write("⭐ Rating:", details.get("vote_average"))

        st.divider()

        # GENRE RECOMMENDATIONS
        st.subheader("🎯 Similar Movies")

        genre_movies = get_genre(mid)

        cols = st.columns(6)
        for i, m in enumerate(genre_movies):
            with cols[i % 6]:
                st.image(m.get("poster_url"), use_container_width=True)
                st.caption(m.get("title"))

        # TF-IDF RECOMMENDATIONS
        st.subheader("🤖 AI Recommendations")

        recs = get_tfidf(details.get("title"))

        for r in recs:
            st.write(f"🎬 {r['title']}  ⭐ {r['score']}")

# ---------------- WATCHLIST ----------------
elif menu == "❤️ Watchlist":

    st.subheader("Your Watchlist ❤️")

    if not st.session_state.watchlist:
        st.info("No movies added yet")
    else:

        cols = st.columns(6)

        for i, m in enumerate(st.session_state.watchlist):
            with cols[i % 6]:
                st.image(m.get("poster_url"), use_container_width=True)
                st.caption(m.get("title"))