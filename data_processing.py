"""
BookWise - Data Processing & PySpark Big Data Pipeline
Processes ~6 million ratings, 10,000 books, and tags using Apache Spark (PySpark).
Calculates big data aggregates, Bayesian popularity scores, genre extraction,
and trains an ALS collaborative filtering latent factor model.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BookWiseDataProcessor")

def setup_environment():
    """Configure Java, Spark, and Python environment paths."""
    workspace = os.path.dirname(os.path.abspath(__file__))
    jdk_workspace = os.path.join(workspace, "jdk17")
    user_profile = os.environ.get("USERPROFILE", "")
    jdk_user = os.path.join(user_profile, "jdk17")

    java_home = None
    if os.path.exists(jdk_workspace):
        java_home = jdk_workspace
    elif os.path.exists(jdk_user):
        java_home = jdk_user

    if java_home:
        os.environ["JAVA_HOME"] = java_home
        java_bin = os.path.join(java_home, "bin")
        if java_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = java_bin + os.pathsep + os.environ.get("PATH", "")
        logger.info(f"Using JAVA_HOME: {java_home}")

    python_exe = sys.executable
    os.environ["PYSPARK_PYTHON"] = python_exe
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_exe
    logger.info(f"Using PySpark Python: {python_exe}")


def get_spark_session(app_name="BookWise-BigData-Analytics"):
    """Initialize an optimized local PySpark SparkSession."""
    setup_environment()
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# Non-genre stop tags to filter out noise from Goodreads tags
STOP_TAGS = {
    "to-read", "currently-reading", "favorites", "default", "owned", 
    "books-i-own", "audiobook", "wish-list", "read", "all-time-favorites", 
    "my-books", "ebook", "kindle", "library", "bought", "series", "audio", 
    "re-read", "to-buy", "ebooks", "favorite", "own", "read-in-2015",
    "read-in-2016", "read-in-2017", "read-in-2014", "read-in-2013",
    "read-in-2012", "audiobooks", "abandoned", "dnf", "did-not-finish",
    "book-club", "shelfari-favorites", "i-own", "have", "borrowed"
}

def run_pyspark_pipeline(data_dir="goodbooks-10k-master", output_dir="data/processed"):
    """
    Executes the end-to-end Big Data PySpark pipeline:
    1. Loads actual CSV datasets (~6M ratings, 10k books, tags).
    2. Cleans nulls and invalid records.
    3. Aggregates book popularity, mean ratings, std dev, and star distributions.
    4. Extracts top genres/tags for each book.
    5. Computes Bayesian Weighted Rating (IMDB formula) for fair ranking.
    6. Trains PySpark ALS (Alternating Least Squares) for collaborative filtering.
    7. Caches & exports structured analytical data and latent factors.
    """
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = "data/cache"
    os.makedirs(cache_dir, exist_ok=True)

    logger.info("Initializing Apache Spark Session...")
    spark = get_spark_session()

    from pyspark.sql import functions as F
    from pyspark.sql.types import IntegerType, FloatType
    from pyspark.sql.window import Window
    from pyspark.ml.recommendation import ALS

    books_csv = os.path.join(data_dir, "books.csv")
    ratings_csv = os.path.join(data_dir, "ratings.csv")
    book_tags_csv = os.path.join(data_dir, "book_tags.csv")
    tags_csv = os.path.join(data_dir, "tags.csv")

    logger.info(f"Loading ratings dataset from {ratings_csv} using PySpark...")
    ratings_df = spark.read.csv(ratings_csv, header=True, inferSchema=True)
    # Clean ratings: ensure ratings are within 1 to 5, drop nulls
    ratings_df = ratings_df.filter(
        (F.col("rating") >= 1) & (F.col("rating") <= 5) & 
        F.col("user_id").isNotNull() & F.col("book_id").isNotNull()
    )

    total_ratings_count = ratings_df.count()
    unique_users_count = ratings_df.select("user_id").distinct().count()
    unique_books_rated = ratings_df.select("book_id").distinct().count()
    logger.info(f"PySpark Processed: {total_ratings_count:,} ratings across {unique_users_count:,} users and {unique_books_rated:,} books.")

    # Rating distribution across 1 to 5 stars using PySpark
    logger.info("Computing star rating distribution with PySpark...")
    rating_distribution = (
        ratings_df.groupBy("rating")
        .count()
        .orderBy("rating")
        .toPandas()
        .to_dict(orient="records")
    )

    # Per-book aggregations using PySpark on 6 million rows
    logger.info("Calculating per-book rating count, mean, and stddev with PySpark...")
    book_stats_df = ratings_df.groupBy("book_id").agg(
        F.count("rating").alias("spark_rating_count"),
        F.round(F.avg("rating"), 3).alias("spark_avg_rating"),
        F.round(F.stddev("rating"), 3).alias("spark_rating_stddev")
    )

    # User activity statistics using PySpark
    logger.info("Calculating user activity metrics with PySpark...")
    user_activity_df = ratings_df.groupBy("user_id").agg(
        F.count("rating").alias("user_rating_count"),
        F.round(F.avg("rating"), 2).alias("user_mean_rating")
    )
    user_activity_stats = user_activity_df.select(
        F.round(F.avg("user_rating_count"), 2).alias("avg_ratings_per_user"),
        F.round(F.min("user_rating_count"), 2).alias("min_ratings_per_user"),
        F.round(F.max("user_rating_count"), 2).alias("max_ratings_per_user"),
        F.percentile_approx("user_rating_count", 0.5).alias("median_ratings_per_user")
    ).toPandas().to_dict(orient="records")[0]

    logger.info("Loading books metadata from books.csv...")
    books_df = (
        spark.read
        .option("header", "true")
        .option("quote", "\"")
        .option("escape", "\"")
        .option("multiLine", "true")
        .csv(books_csv)
    )

    # Cast schema safely
    books_df = books_df.select(
        F.col("book_id").cast(FloatType()).cast(IntegerType()).alias("book_id"),
        F.col("goodreads_book_id").cast(FloatType()).cast(IntegerType()).alias("goodreads_book_id"),
        F.coalesce(F.col("authors"), F.lit("Unknown Author")).alias("authors"),
        F.coalesce(F.col("original_publication_year").cast(FloatType()).cast(IntegerType()), F.lit(0)).alias("original_publication_year"),
        F.coalesce(F.col("original_title"), F.col("title")).alias("original_title"),
        F.coalesce(F.col("title"), F.lit("Unknown Title")).alias("title"),
        F.coalesce(F.col("language_code"), F.lit("unknown")).alias("language_code"),
        F.coalesce(F.col("average_rating").cast(FloatType()), F.lit(0.0)).alias("average_rating"),
        F.coalesce(F.col("ratings_count").cast(FloatType()).cast(IntegerType()), F.lit(0)).alias("ratings_count"),
        F.coalesce(F.col("image_url"), F.lit("")).alias("image_url"),
        F.coalesce(F.col("small_image_url"), F.lit("")).alias("small_image_url")
    ).filter(F.col("book_id").isNotNull())

    # Join books with Spark-calculated ratings statistics
    joined_books = books_df.join(book_stats_df, on="book_id", how="left")

    # Bayesian Weighted Rating Formula:
    # WR = (v / (v + m)) * R + (m / (v + m)) * C
    # v = book's ratings count
    # R = book's average rating
    # m = minimum threshold (500 ratings)
    # C = mean rating across all books (~3.92)
    mean_rating_all = ratings_df.select(F.avg("rating")).collect()[0][0]
    min_ratings_threshold = 500.0

    joined_books = joined_books.withColumn(
        "v", F.coalesce(F.col("spark_rating_count").cast(FloatType()), F.col("ratings_count").cast(FloatType()))
    ).withColumn(
        "R", F.coalesce(F.col("spark_avg_rating").cast(FloatType()), F.col("average_rating").cast(FloatType()))
    ).withColumn(
        "weighted_rating",
        F.round(
            (F.col("v") / (F.col("v") + F.lit(min_ratings_threshold))) * F.col("R") +
            (F.lit(min_ratings_threshold) / (F.col("v") + F.lit(min_ratings_threshold))) * F.lit(mean_rating_all),
            3
        )
    ).drop("v", "R")

    # Process Book Tags using PySpark
    logger.info("Processing book tags and extracting top genres with PySpark...")
    book_tags_df = spark.read.csv(book_tags_csv, header=True, inferSchema=True)
    tags_df = spark.read.csv(tags_csv, header=True, inferSchema=True)

    # Filter out stop tags
    meaningful_tags = tags_df.filter(~F.col("tag_name").isin(list(STOP_TAGS)))

    bt_joined = book_tags_df.join(meaningful_tags, on="tag_id", how="inner")
    
    # Rank top 6 tags per goodreads_book_id
    window_spec = Window.partitionBy("goodreads_book_id").orderBy(F.col("count").desc())
    ranked_tags = bt_joined.withColumn("rank", F.row_number().over(window_spec)).filter(F.col("rank") <= 6)

    # Aggregate tags into a comma-separated string per book
    book_genres_df = ranked_tags.groupBy("goodreads_book_id").agg(
        F.concat_ws(", ", F.collect_list("tag_name")).alias("genres")
    )

    # Join genres back to books
    books_with_genres = joined_books.join(book_genres_df, on="goodreads_book_id", how="left")
    books_with_genres = books_with_genres.na.fill({"genres": "Fiction, Literature"})

    # Convert processed books to Pandas for storage and fast querying
    logger.info("Exporting processed books dataset...")
    pandas_books = books_with_genres.toPandas()

    # Fill NaN values in pandas DataFrame
    pandas_books["genres"] = pandas_books["genres"].fillna("Fiction, Literature")
    pandas_books["original_title"] = pandas_books["original_title"].fillna(pandas_books["title"])
    pandas_books["original_publication_year"] = pandas_books["original_publication_year"].fillna(0).astype(int)
    pandas_books["weighted_rating"] = pandas_books["weighted_rating"].fillna(pandas_books["average_rating"])

    books_output_path = os.path.join(output_dir, "books_processed.csv")
    pandas_books.to_csv(books_output_path, index=False)
    logger.info(f"Processed books saved to: {books_output_path}")

    # Top authors analysis using PySpark
    logger.info("Analyzing top authors with PySpark...")
    top_authors_ratings = (
        joined_books.groupBy("authors")
        .agg(
            F.count("book_id").alias("total_books"),
            F.sum("ratings_count").alias("total_ratings"),
            F.round(F.avg("average_rating"), 2).alias("avg_rating")
        )
        .filter(F.col("total_books") >= 2)
        .orderBy(F.col("total_ratings").desc())
        .limit(20)
        .toPandas()
        .to_dict(orient="records")
    )

    # Train PySpark ALS Collaborative Filtering Model
    logger.info("Training PySpark ALS Collaborative Filtering Model on 6M ratings...")
    als = ALS(
        userCol="user_id",
        itemCol="book_id",
        ratingCol="rating",
        rank=12,
        maxIter=6,
        regParam=0.1,
        coldStartStrategy="drop",
        nonnegative=True,
        seed=42
    )

    als_model = als.fit(ratings_df)
    logger.info("ALS model training completed successfully.")

    # Extract item latent factors
    item_factors_df = als_model.itemFactors
    pandas_item_factors = item_factors_df.toPandas()

    # Expand feature vector list to separate columns
    factors_matrix = np.array(pandas_item_factors["features"].tolist())
    item_ids = pandas_item_factors["id"].values
    factors_df = pd.DataFrame(factors_matrix)
    factors_df["book_id"] = item_ids

    factors_output_path = os.path.join(output_dir, "item_factors.csv")
    factors_df.to_csv(factors_output_path, index=False)
    logger.info(f"ALS item latent factors saved to: {factors_output_path}")

    # Compile Analytics Summary
    analytics_summary = {
        "total_ratings": int(total_ratings_count),
        "total_users": int(unique_users_count),
        "total_books": int(unique_books_rated),
        "global_mean_rating": round(float(mean_rating_all), 3),
        "rating_distribution": rating_distribution,
        "user_activity": user_activity_stats,
        "top_authors": top_authors_ratings,
        "engine_status": "PySpark Big Data Engine Active"
    }

    summary_path = os.path.join(output_dir, "analytics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(analytics_summary, f, indent=2)
    logger.info(f"Analytics summary saved to: {summary_path}")

    spark.stop()
    logger.info("PySpark pipeline execution finished successfully!")
    return pandas_books, analytics_summary


def load_or_process_data(force_reprocess=False):
    """
    Checks if cached processed data is available.
    If not, or if force_reprocess is True, runs the PySpark pipeline.
    """
    processed_books_path = "data/processed/books_processed.csv"
    summary_path = "data/processed/analytics_summary.json"
    factors_path = "data/processed/item_factors.csv"

    if not force_reprocess and os.path.exists(processed_books_path) and os.path.exists(summary_path) and os.path.exists(factors_path):
        logger.info("Loading cached processed data...")
        books_df = pd.read_csv(processed_books_path)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        item_factors = pd.read_csv(factors_path)
        return books_df, summary, item_factors

    logger.info("Cached processed data not found or reprocess requested. Running PySpark pipeline...")
    books_df, summary = run_pyspark_pipeline()
    item_factors = pd.read_csv(factors_path)
    return books_df, summary, item_factors


if __name__ == "__main__":
    force = "--reprocess" in sys.argv
    load_or_process_data(force_reprocess=force)
