"""
BookWise - Recommendation Engine
Implements:
1. Content-Based Filtering (TF-IDF + Cosine Similarity on title, authors, genres/tags)
2. Collaborative Filtering (Latent Item Factors from PySpark ALS Model)
3. Hybrid Recommendation (Content + Collaborative blending)
4. Popularity-Based Recommendations (Bayesian Weighted Rating)
Strict Rule: Never recommend the selected book itself.
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

class BookRecommendationEngine:
    def __init__(self, processed_data_dir="data/processed"):
        self.processed_data_dir = processed_data_dir
        self.books_df = None
        self.item_factors_df = None
        self.tfidf_matrix = None
        self.vectorizer = None
        self.book_id_to_idx = {}
        self.idx_to_book_id = {}
        self.als_matrix = None
        self.als_book_id_to_row = {}
        self._load_and_initialize()

    def _load_and_initialize(self):
        """Loads processed books and builds similarity index."""
        books_path = os.path.join(self.processed_data_dir, "books_processed.csv")
        factors_path = os.path.join(self.processed_data_dir, "item_factors.csv")

        if not os.path.exists(books_path):
            raise FileNotFoundError(
                f"Processed data not found at {books_path}. Run data_processing.py first!"
            )

        self.books_df = pd.read_csv(books_path)
        # Clean potential nulls
        self.books_df["title"] = self.books_df["title"].fillna("Unknown Title")
        self.books_df["authors"] = self.books_df["authors"].fillna("Unknown Author")
        self.books_df["genres"] = self.books_df["genres"].fillna("Fiction")
        self.books_df["image_url"] = self.books_df["image_url"].fillna("")
        self.books_df["average_rating"] = self.books_df["average_rating"].fillna(0.0)
        self.books_df["ratings_count"] = self.books_df["ratings_count"].fillna(0)
        self.books_df["weighted_rating"] = self.books_df["weighted_rating"].fillna(
            self.books_df["average_rating"]
        )

        # Build index maps
        self.book_id_to_idx = {
            int(bid): idx for idx, bid in enumerate(self.books_df["book_id"])
        }
        self.idx_to_book_id = {
            idx: int(bid) for idx, bid in enumerate(self.books_df["book_id"])
        }

        # Build Content-Based Metadata Soup: Title + Author (weighted x2) + Genres (weighted x2)
        # Repeating author and genres gives higher similarity weighting to same author / genre
        metadata_soup = (
            self.books_df["title"] + " " +
            self.books_df["authors"] + " " + self.books_df["authors"] + " " +
            self.books_df["genres"] + " " + self.books_df["genres"]
        )

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=1,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(metadata_soup)

        # Load PySpark ALS Item Factors if available
        if os.path.exists(factors_path):
            self.item_factors_df = pd.read_csv(factors_path)
            factor_cols = [c for c in self.item_factors_df.columns if c != "book_id"]
            self.als_matrix = self.item_factors_df[factor_cols].values
            
            # Normalize ALS factor vectors for cosine similarity
            norms = np.linalg.norm(self.als_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            self.als_matrix = self.als_matrix / norms

            for row_idx, bid in enumerate(self.item_factors_df["book_id"]):
                self.als_book_id_to_row[int(bid)] = row_idx

    def get_book_by_id(self, book_id: int):
        """Returns book dict by book_id."""
        book_id = int(book_id)
        match = self.books_df[self.books_df["book_id"] == book_id]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    def search_books(self, query: str, limit: int = 15):
        """Searches books by title, author, or genre substring."""
        if not query or not query.strip():
            return self.books_df.head(limit).to_dict(orient="records")

        q = query.strip().lower()
        mask = (
            self.books_df["title"].str.lower().str.contains(q, na=False) |
            self.books_df["authors"].str.lower().str.contains(q, na=False) |
            self.books_df["genres"].str.lower().str.contains(q, na=False)
        )
        matches = self.books_df[mask].sort_values(by="weighted_rating", ascending=False)
        return matches.head(limit).to_dict(orient="records")

    def get_popular_books(self, n: int = 10, genre_filter: str = None):
        """
        Returns top N popular books ranked by Bayesian Weighted Rating.
        Supports optional genre filtering.
        """
        df = self.books_df
        if genre_filter and genre_filter.strip() and genre_filter.lower() != "all":
            gf = genre_filter.strip().lower()
            df = df[df["genres"].str.lower().str.contains(gf, na=False)]

        top_books = df.sort_values(by="weighted_rating", ascending=False).head(n)
        return top_books.to_dict(orient="records")

    def get_content_based_recommendations(self, target_book_id: int, n: int = 8):
        """
        Content-Based Filtering:
        Uses TF-IDF vector cosine similarity across book titles, authors, and genres.
        Guaranteed: Never recommends the target book itself!
        """
        target_book_id = int(target_book_id)
        if target_book_id not in self.book_id_to_idx:
            return []

        target_idx = self.book_id_to_idx[target_book_id]
        target_vec = self.tfidf_matrix[target_idx]

        # Calculate cosine similarity of target with all books
        cosine_sim = linear_kernel(target_vec, self.tfidf_matrix).flatten()

        # Sort indices in descending order
        similar_indices = np.argsort(cosine_sim)[::-1]

        recommendations = []
        for idx in similar_indices:
            candidate_book_id = self.idx_to_book_id[idx]
            # Strict constraint: Never recommend the selected book itself
            if candidate_book_id == target_book_id:
                continue

            book_dict = self.books_df.iloc[idx].to_dict()
            book_dict["similarity_score"] = round(float(cosine_sim[idx]), 3)
            book_dict["recommendation_type"] = "Content-Based"
            recommendations.append(book_dict)

            if len(recommendations) >= n:
                break

        return recommendations

    def get_collaborative_recommendations(self, target_book_id: int, n: int = 8):
        """
        Collaborative Filtering:
        Computes cosine similarity in the PySpark ALS latent factor space.
        Represents implicit user co-rating patterns across ~6 million ratings.
        Guaranteed: Never recommends the target book itself!
        """
        target_book_id = int(target_book_id)
        if self.als_matrix is None or target_book_id not in self.als_book_id_to_row:
            # Fallback to content-based if latent factor not found
            return self.get_content_based_recommendations(target_book_id, n=n)

        target_row = self.als_book_id_to_row[target_book_id]
        target_factor = self.als_matrix[target_row]

        # Matrix dot product (already normalized)
        sim_scores = np.dot(self.als_matrix, target_factor)
        sorted_rows = np.argsort(sim_scores)[::-1]

        recommendations = []
        for r in sorted_rows:
            candidate_book_id = int(self.item_factors_df.iloc[r]["book_id"])
            # Strict constraint: Never recommend the selected book itself
            if candidate_book_id == target_book_id:
                continue

            book_data = self.get_book_by_id(candidate_book_id)
            if not book_data:
                continue

            book_data["similarity_score"] = round(float(sim_scores[r]), 3)
            book_data["recommendation_type"] = "Collaborative (PySpark ALS)"
            recommendations.append(book_data)

            if len(recommendations) >= n:
                break

        return recommendations

    def get_hybrid_recommendations(self, target_book_id: int, n: int = 8, content_weight: float = 0.5):
        """
        Hybrid Filtering:
        Blends Content-Based TF-IDF scores with PySpark ALS Collaborative Filtering latent scores.
        Score = w_content * S_content + (1 - w_content) * S_collaborative
        Guaranteed: Never recommends the target book itself!
        """
        target_book_id = int(target_book_id)
        if target_book_id not in self.book_id_to_idx:
            return []

        target_idx = self.book_id_to_idx[target_book_id]
        # Content scores
        target_vec = self.tfidf_matrix[target_idx]
        content_sim = linear_kernel(target_vec, self.tfidf_matrix).flatten()

        # Collaborative scores
        collab_sim = np.zeros(len(self.books_df))
        if self.als_matrix is not None and target_book_id in self.als_book_id_to_row:
            target_row = self.als_book_id_to_row[target_book_id]
            target_factor = self.als_matrix[target_row]
            als_sims = np.dot(self.als_matrix, target_factor)
            
            # Map ALS similarities back to books_df row indices
            for r, sim in enumerate(als_sims):
                b_id = int(self.item_factors_df.iloc[r]["book_id"])
                if b_id in self.book_id_to_idx:
                    idx = self.book_id_to_idx[b_id]
                    collab_sim[idx] = max(0.0, float(sim))

        # Normalize collab scores to [0, 1]
        collab_max = np.max(collab_sim)
        if collab_max > 0:
            collab_sim = collab_sim / collab_max

        # Blend
        hybrid_scores = content_weight * content_sim + (1.0 - content_weight) * collab_sim

        # Sort indices
        sorted_indices = np.argsort(hybrid_scores)[::-1]

        recommendations = []
        for idx in sorted_indices:
            candidate_book_id = self.idx_to_book_id[idx]
            # Strict constraint: Never recommend the selected book itself
            if candidate_book_id == target_book_id:
                continue

            book_dict = self.books_df.iloc[idx].to_dict()
            book_dict["similarity_score"] = round(float(hybrid_scores[idx]), 3)
            book_dict["content_score"] = round(float(content_sim[idx]), 3)
            book_dict["collab_score"] = round(float(collab_sim[idx]), 3)
            book_dict["recommendation_type"] = "Hybrid (Content + ALS)"
            recommendations.append(book_dict)

            if len(recommendations) >= n:
                break

        return recommendations

