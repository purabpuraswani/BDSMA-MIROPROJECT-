from recommendation import BookRecommendationEngine

engine = BookRecommendationEngine()
print(f"Total Books Loaded: {len(engine.books_df)}")

# Test Book 1: The Hunger Games
target_id = 1
book = engine.get_book_by_id(target_id)
print(f"\nTarget Book: {book['title']} by {book['authors']} (Genres: {book['genres']})")

print("\n--- 1. Content-Based Recommendations ---")
content_recs = engine.get_content_based_recommendations(target_id, n=5)
for r in content_recs:
    assert r['book_id'] != target_id, "ERROR: Target book was recommended!"
    print(f"[{r['book_id']}] {r['title']} by {r['authors']} (Score: {r['similarity_score']})")

print("\n--- 2. Collaborative Filtering (PySpark ALS) Recommendations ---")
collab_recs = engine.get_collaborative_recommendations(target_id, n=5)
for r in collab_recs:
    assert r['book_id'] != target_id, "ERROR: Target book was recommended!"
    print(f"[{r['book_id']}] {r['title']} by {r['authors']} (Score: {r['similarity_score']})")

print("\n--- 3. Hybrid Recommendations ---")
hybrid_recs = engine.get_hybrid_recommendations(target_id, n=5)
for r in hybrid_recs:
    assert r['book_id'] != target_id, "ERROR: Target book was recommended!"
    print(f"[{r['book_id']}] {r['title']} by {r['authors']} (Hybrid Score: {r['similarity_score']})")

print("\n--- 4. Popular Books (Bayesian Weighted Rating) ---")
pop_recs = engine.get_popular_books(n=5)
for r in pop_recs:
    print(f"[{r['book_id']}] {r['title']} by {r['authors']} (WR: {r['weighted_rating']} | Avg: {r['average_rating']})")

print("\n--- 5. Genre Filtered Popular Books (Fantasy) ---")
fantasy_recs = engine.get_popular_books(n=5, genre_filter="fantasy")
for r in fantasy_recs:
    print(f"[{r['book_id']}] {r['title']} by {r['authors']} (Genres: {r['genres']})")

print("\nAll recommendation tests PASSED successfully!")

