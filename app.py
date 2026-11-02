"""
BookWise - A Big Data-Driven Personalized Book Recommendation System
Built with Streamlit, Apache Spark (PySpark), Scikit-Learn, and Plotly.
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configure page settings
st.set_page_config(
    page_title="BookWise | Big Data Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Gradient header styling */
    .main-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.6rem;
        margin-bottom: 0.2rem;
    }
    
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 1.8rem;
    }

    /* Stat card */
    .stat-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-3px);
        border-color: #6366f1;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        color: #f8fafc;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
        margin-top: 0.3rem;
    }

    /* Book Card */
    .book-card-container {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 12px;
        margin-bottom: 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }
    .book-card-container:hover {
        transform: translateY(-5px);
        border-color: #8b5cf6;
        box-shadow: 0 20px 25px -5px rgba(139, 92, 246, 0.25);
    }
    .book-cover {
        width: 100%;
        height: 220px;
        object-fit: cover;
        border-radius: 8px;
        background: #0f172a;
        margin-bottom: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.4);
    }
    .book-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #f1f5f9;
        line-height: 1.3;
        margin-bottom: 4px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        height: 2.6em;
    }
    .book-author {
        font-size: 0.82rem;
        color: #cbd5e1;
        margin-bottom: 6px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .badge-rating {
        display: inline-block;
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 2px 7px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .badge-sim {
        display: inline-block;
        background: rgba(139, 92, 246, 0.18);
        color: #c084fc;
        border: 1px solid rgba(139, 92, 246, 0.35);
        padding: 2px 7px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .genre-tag {
        display: inline-block;
        background: #334155;
        color: #94a3b8;
        font-size: 0.7rem;
        padding: 2px 6px;
        border-radius: 4px;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    .badge-spark {
        background: linear-gradient(90deg, #e11d48, #ea580c);
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_recommendation_engine():
    """Initializes and caches the BookWise Recommendation Engine."""
    from recommendation import BookRecommendationEngine
    return BookRecommendationEngine()


@st.cache_data
def get_analytics_summary():
    """Loads pre-calculated PySpark big data analytics summary."""
    summary_path = "data/processed/analytics_summary.json"
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# Helper to render a consistent book cover
def get_cover_image_html(image_url, title):
    if image_url and isinstance(image_url, str) and image_url.strip() and image_url.startswith("http"):
        return f'<img src="{image_url}" class="book-cover" alt="{title}" onerror="this.onerror=null;this.src=\'https://via.placeholder.com/150x220/1e293b/94a3b8?text=BookWise\';">'
    return '<div class="book-cover" style="display:flex;align-items:center;justify-content:center;color:#94a3b8;font-size:0.8rem;text-align:center;padding:10px;">📖 No Cover</div>'


def render_book_card(book, extra_badge=None):
    title = book.get("title", "Unknown Title")
    authors = book.get("authors", "Unknown Author")
    avg_rating = round(float(book.get("average_rating", 0.0)), 2)
    img_url = book.get("image_url", "")
    genres = str(book.get("genres", "")).split(",")[:2]
    tags_html = "".join([f'<span class="genre-tag">{g.strip().capitalize()}</span>' for g in genres if g.strip()])

    badge_html = f'<span class="badge-rating">★ {avg_rating}</span>'
    if extra_badge:
        badge_html += f' <span class="badge-sim">{extra_badge}</span>'

    card_html = f"""
    <div class="book-card-container">
        {get_cover_image_html(img_url, title)}
        <div>
            <div class="book-title" title="{title}">{title}</div>
            <div class="book-author" title="{authors}">by {authors}</div>
            <div style="margin-bottom: 8px;">{badge_html}</div>
            <div style="height: 26px; overflow: hidden;">{tags_html}</div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


# Main App Initialization
try:
    engine = get_recommendation_engine()
    summary = get_analytics_summary()
except Exception as e:
    st.error(f"Error loading BookWise data: {e}. Please ensure data_processing.py has completed.")
    st.stop()


# Sidebar Navigation
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1495446815901-a7297e633e8d?w=400&auto=format&fit=crop&q=60", use_container_width=True)
    st.markdown("## **BookWise Engine**")
    st.markdown('<span class="badge-spark">⚡ Powered by Apache Spark</span>', unsafe_allow_html=True)
    st.markdown("---")

    menu = st.radio(
        "Navigation",
        [
            "🏠 Discover & Popular",
            "🎯 Smart Recommendations",
            "🔍 Search & Explore",
            "📊 Big Data Analytics",
            "⚡ Spark Architecture"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### 📌 **Microproject Info**")
    st.caption("**Course:** Big Data Analytics (BDSMA)")
    st.caption("**Dataset:** Goodbooks-10k (~6M ratings)")
    st.caption("**Pipeline:** PySpark 4.3 + Scikit-Learn")
    st.caption("**Model:** ALS Latent Factors + TF-IDF")


# -------------------------------------------------------------
# 1. DISCOVER & POPULAR
# -------------------------------------------------------------
if menu == "🏠 Discover & Popular":
    st.markdown('<div class="main-header">BookWise</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">A Big Data-Driven Personalized Book Recommendation Engine powered by Apache Spark</div>', unsafe_allow_html=True)

    # Top KPI Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{summary.get('total_books', 10000):,}</div>
            <div class="stat-label">Total Books Cataloged</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{summary.get('total_ratings', 5976479):,}</div>
            <div class="stat-label">PySpark Processed Ratings</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{summary.get('total_users', 53424):,}</div>
            <div class="stat-label">Active Readers Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">★ {summary.get('global_mean_rating', 3.92)}</div>
            <div class="stat-label">Global Mean Rating</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Popular Books Section
    col_hdr, col_flt = st.columns([3, 1])
    with col_hdr:
        st.subheader("🔥 Trending & Highest-Rated Books")
        st.caption("Ranked using PySpark Bayesian Weighted Rating (IMDB Bayesian Formula)")
    with col_flt:
        genre_choice = st.selectbox(
            "Filter by Genre",
            ["All", "Fantasy", "Fiction", "Classics", "Young-Adult", "Sci-Fi", "Mystery", "Romance", "Nonfiction"]
        )

    popular_books = engine.get_popular_books(n=10, genre_filter=genre_choice)

    # Render in 5 columns x 2 rows
    cols = st.columns(5)
    for i, book in enumerate(popular_books[:5]):
        with cols[i]:
            wr = round(float(book.get("weighted_rating", 0.0)), 2)
            render_book_card(book, extra_badge=f"WR: {wr}")
            if st.button("Recommend", key=f"rec_pop_{book['book_id']}"):
                st.session_state["selected_book_id"] = int(book["book_id"])
                st.session_state["nav_to_recs"] = True
                st.rerun()

    cols2 = st.columns(5)
    for i, book in enumerate(popular_books[5:10]):
        with cols2[i]:
            wr = round(float(book.get("weighted_rating", 0.0)), 2)
            render_book_card(book, extra_badge=f"WR: {wr}")
            if st.button("Recommend", key=f"rec_pop2_{book['book_id']}"):
                st.session_state["selected_book_id"] = int(book["book_id"])
                st.session_state["nav_to_recs"] = True
                st.rerun()


# -------------------------------------------------------------
# 2. SMART RECOMMENDATIONS
# -------------------------------------------------------------
elif menu == "🎯 Smart Recommendations":
    st.markdown('<div class="main-header">Personalized Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Select a book to generate tailored recommendations using Content-Based, PySpark Collaborative, or Hybrid algorithms.</div>', unsafe_allow_html=True)

    # Book selector
    books_list = engine.books_df[["book_id", "title", "authors"]].copy()
    books_list["display_name"] = books_list["title"] + " — " + books_list["authors"]
    book_options = dict(zip(books_list["book_id"], books_list["display_name"]))

    # Default selection
    default_id = st.session_state.get("selected_book_id", 1)
    if default_id not in book_options:
        default_id = 1

    col_sel, col_alg, col_cnt = st.columns([3, 2, 1])

    with col_sel:
        selected_book_id = st.selectbox(
            "Select or Search a Seed Book:",
            options=list(book_options.keys()),
            format_func=lambda x: book_options.get(x, f"Book {x}"),
            index=list(book_options.keys()).index(default_id)
        )

    with col_alg:
        algo_choice = st.selectbox(
            "Recommendation Strategy:",
            [
                "🚀 Hybrid (Content + Spark ALS)",
                "🧠 Content-Based (TF-IDF & Tags)",
                "🤝 Collaborative (PySpark ALS Latent Factors)"
            ]
        )

    with col_cnt:
        rec_count = st.slider("Results Count:", min_value=5, max_value=10, value=8)

    # Show Selected Book Summary
    seed_book = engine.get_book_by_id(selected_book_id)
    if seed_book:
        st.markdown("---")
        card_col1, card_col2 = st.columns([1, 4])
        with card_col1:
            st.markdown(get_cover_image_html(seed_book.get("image_url", ""), seed_book.get("title")), unsafe_allow_html=True)
        with card_col2:
            st.markdown(f"### **{seed_book.get('title')}**")
            st.markdown(f"**Author(s):** {seed_book.get('authors')}")
            st.markdown(f"**Published Year:** {int(seed_book.get('original_publication_year', 0))} | **Average Rating:** ★ {seed_book.get('average_rating', 0.0)} ({int(seed_book.get('ratings_count', 0)):,} ratings)")
            st.markdown(f"**Bayesian Weighted Score:** {seed_book.get('weighted_rating', 0.0)}")
            genres = str(seed_book.get("genres", "")).split(",")
            pill_tags = " ".join([f"`{g.strip().capitalize()}`" for g in genres if g.strip()])
            st.markdown(f"**Genre Keywords:** {pill_tags}")

        st.markdown("---")

        # Generate recommendations based on strategy
        st.subheader(f"Recommendations based on *'{seed_book.get('title')}'*")
        
        if "Content-Based" in algo_choice:
            recs = engine.get_content_based_recommendations(selected_book_id, n=rec_count)
            st.caption("ℹ️ Computed via TF-IDF vectorization and cosine similarity over author, title, and book tag spaces.")
        elif "Collaborative" in algo_choice:
            recs = engine.get_collaborative_recommendations(selected_book_id, n=rec_count)
            st.caption("ℹ️ Computed via PySpark ALS Matrix Factorization item latent embeddings trained on ~6M ratings.")
        else:
            recs = engine.get_hybrid_recommendations(selected_book_id, n=rec_count)
            st.caption("ℹ️ Blended scoring: 50% Content Similarity + 50% PySpark ALS Collaborative Filtering.")

        # Ensure selected book is never recommended itself
        recs = [r for r in recs if int(r["book_id"]) != int(selected_book_id)]

        if not recs:
            st.warning("No recommendations found for the selected book.")
        else:
            # Display in responsive grid
            grid_cols = st.columns(5)
            for idx, rbook in enumerate(recs):
                col_target = grid_cols[idx % 5]
                with col_target:
                    sim_pct = f"{int(rbook.get('similarity_score', 0) * 100)}% Match"
                    render_book_card(rbook, extra_badge=sim_pct)
                    if st.button("Pivot to this", key=f"pivot_{rbook['book_id']}"):
                        st.session_state["selected_book_id"] = int(rbook["book_id"])
                        st.rerun()


# -------------------------------------------------------------
# 3. SEARCH & EXPLORE
# -------------------------------------------------------------
elif menu == "🔍 Search & Explore":
    st.markdown('<div class="main-header">Book Catalog & Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Search across 10,000 titles, authors, and genres with instant filters.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search_query = st.text_input("Search by title, author, or keyword:", placeholder="e.g. Harry Potter, Agatha Christie, Dystopian, Tolkien...")
    with col2:
        min_rating = st.slider("Min Rating:", 1.0, 5.0, 3.8, 0.1)
    with col3:
        sort_by = st.selectbox("Sort By:", ["Popularity (Weighted)", "Average Rating", "Total Reviews", "Publication Year"])

    filtered_books = engine.books_df.copy()
    if search_query and search_query.strip():
        q = search_query.strip().lower()
        filtered_books = filtered_books[
            filtered_books["title"].str.lower().str.contains(q, na=False) |
            filtered_books["authors"].str.lower().str.contains(q, na=False) |
            filtered_books["genres"].str.lower().str.contains(q, na=False)
        ]

    filtered_books = filtered_books[filtered_books["average_rating"] >= min_rating]

    if sort_by == "Popularity (Weighted)":
        filtered_books = filtered_books.sort_values(by="weighted_rating", ascending=False)
    elif sort_by == "Average Rating":
        filtered_books = filtered_books.sort_values(by="average_rating", ascending=False)
    elif sort_by == "Total Reviews":
        filtered_books = filtered_books.sort_values(by="ratings_count", ascending=False)
    else:
        filtered_books = filtered_books.sort_values(by="original_publication_year", ascending=False)

    st.markdown(f"**Found {len(filtered_books):,} matching books**")

    # Display results
    display_limit = min(20, len(filtered_books))
    results = filtered_books.head(display_limit).to_dict(orient="records")

    cols = st.columns(5)
    for i, book in enumerate(results):
        with cols[i % 5]:
            render_book_card(book)
            if st.button("Explore Similar", key=f"search_btn_{book['book_id']}"):
                st.session_state["selected_book_id"] = int(book["book_id"])
                st.session_state["nav_to_recs"] = True
                st.rerun()


# -------------------------------------------------------------
# 4. BIG DATA ANALYTICS
# -------------------------------------------------------------
elif menu == "📊 Big Data Analytics":
    st.markdown('<div class="main-header">PySpark Big Data Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive insights from 5,976,479 reader ratings aggregated with Apache Spark.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["⭐ Rating Patterns", "✍️ Top Authors", "👥 User Engagement"])

    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Star Rating Distribution (1–5 Stars)")
            dist_data = summary.get("rating_distribution", [])
            dist_df = pd.DataFrame(dist_data)
            
            fig_stars = px.bar(
                dist_df,
                x="rating",
                y="count",
                labels={"rating": "Star Rating", "count": "Total Votes"},
                color="rating",
                color_continuous_scale="Purples",
                title="Total Ratings per Star Level"
            )
            fig_stars.update_layout(showlegend=False, template="plotly_dark", height=380)
            st.plotly_chart(fig_stars, use_container_width=True)

        with c2:
            st.subheader("Top 10 Highest-Rated Books (Bayesian Weighted)")
            top_rated = engine.books_df.sort_values(by="weighted_rating", ascending=False).head(10)
            fig_top = px.bar(
                top_rated,
                x="weighted_rating",
                y="title",
                orientation="h",
                labels={"weighted_rating": "Bayesian Rating", "title": "Book Title"},
                color="weighted_rating",
                color_continuous_scale="Viridis",
                title="Top 10 Books by Weighted Popularity"
            )
            fig_top.update_layout(yaxis={"autorange": "reversed"}, template="plotly_dark", height=380)
            st.plotly_chart(fig_top, use_container_width=True)

    with tab2:
        st.subheader("Most Influential Authors in Dataset (by Total Ratings & Works)")
        authors_data = summary.get("top_authors", [])
        authors_df = pd.DataFrame(authors_data)

        if not authors_df.empty:
            c_auth1, c_auth2 = st.columns([3, 2])
            with c_auth1:
                fig_author = px.bar(
                    authors_df.head(10),
                    x="total_ratings",
                    y="authors",
                    orientation="h",
                    color="avg_rating",
                    color_continuous_scale="Sunset",
                    labels={"total_ratings": "Total PySpark Ratings", "authors": "Author", "avg_rating": "Avg Rating"},
                    title="Top 10 Authors by Total Rating Volume"
                )
                fig_author.update_layout(yaxis={"autorange": "reversed"}, template="plotly_dark", height=420)
                st.plotly_chart(fig_author, use_container_width=True)

            with c_auth2:
                fig_scatter = px.scatter(
                    authors_df,
                    x="total_books",
                    y="avg_rating",
                    size="total_ratings",
                    hover_name="authors",
                    color="avg_rating",
                    color_continuous_scale="Plasma",
                    labels={"total_books": "Number of Books in 10k", "avg_rating": "Avg Rating"},
                    title="Author Catalog Size vs Average Rating"
                )
                fig_scatter.update_layout(template="plotly_dark", height=420)
                st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        st.subheader("Reader Engagement & Community Dynamics")
        u_stats = summary.get("user_activity", {})
        
        uc1, uc2, uc3, uc4 = st.columns(4)
        with uc1:
            st.metric("Mean Ratings / User", f"{u_stats.get('avg_ratings_per_user', 0)}")
        with uc2:
            st.metric("Median Ratings / User", f"{u_stats.get('median_ratings_per_user', 0)}")
        with uc3:
            st.metric("Max Ratings by Power User", f"{int(u_stats.get('max_ratings_per_user', 0)):,}")
        with uc4:
            st.metric("Min Ratings / User", f"{int(u_stats.get('min_ratings_per_user', 0))}")

        st.info("💡 **Big Data Insight:** The user rating distribution follows a classic long-tail power law distribution, which makes Alternating Least Squares (ALS) matrix factorization well-suited for collaborative filtering by overcoming sparsity in user-item interactions.")


# -------------------------------------------------------------
# 5. SPARK ARCHITECTURE
# -------------------------------------------------------------
elif menu == "⚡ Spark Architecture":
    st.markdown('<div class="main-header">Big Data Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Engine design and distributed processing pipeline workflow.</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🏛️ **System Architecture & Data Flow**
    ```
    [ goodbooks-10k CSVs ] (ratings.csv ~6M rows, books.csv 10k, tags.csv)
             │
             ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                Apache Spark (PySpark 4.3) Pipeline           │
    │                                                              │
    │  1. Ingestion: Distributed CSV reading with schema casting   │
    │  2. Data Cleansing: Null removal, rating range (1-5) bounds  │
    │  3. Aggregation: Per-book mean & variance, user activity     │
    │  4. Bayesian Scoring: WR = (v/(v+m))*R + (m/(v+m))*C         │
    │  5. Tag Mining: Window rank join on 1M tags to extract genre │
    │  6. ALS Model: Distributed Matrix Factorization (rank=12)    │
    └──────────────────────────────────────────────────────────────┘
             │
             ├──► [ Caching Layer ] (books_processed.csv, item_factors.csv)
             │
             ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                Recommendation Engine Layer                   │
    │                                                              │
    │  • Content-Based: TF-IDF Vectorizer + Cosine Similarity      │
    │  • Collaborative: Cosine similarity on Spark ALS item factors│
    │  • Hybrid: Weighted fusion (0.5 Content + 0.5 Collab)        │
    │  • Self-Filter: Strict exclusion of target book              │
    └──────────────────────────────────────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────────────────────────────┐
    │             Streamlit Web Application & Plotly UI            │
    └──────────────────────────────────────────────────────────────┘
    ```
    """)

    st.markdown("### ⚙️ **PySpark Live Pipeline Execution**")
    st.markdown("You can trigger a full PySpark recomputation of the 6-million rating dataset below:")

    if st.button("🔄 Trigger PySpark Reprocessing Job", type="primary"):
        with st.spinner("Running PySpark 4.3 distributed pipeline on goodbooks-10k..."):
            from data_processing import run_pyspark_pipeline
            run_pyspark_pipeline()
            st.success("PySpark pipeline execution finished successfully! Cache updated.")
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

