# BookWise 📚
### A Big Data-Driven Personalized Book Recommendation System
**Big Data Analytics Microproject**

---

## 🌟 Executive Summary
**BookWise** is an enterprise-grade book discovery and personalized recommendation platform powered by **Apache Spark (PySpark)** and **Scikit-Learn**, built on top of the comprehensive **Goodbooks-10k** dataset containing **5,976,479 reader ratings**, **10,000 books**, **53,424 unique users**, and over **1,000,000 book tags**.

BookWise combines distributed big data aggregation, machine learning, and an interactive modern web dashboard to deliver high-precision book recommendations, popularity rankings, and large-scale analytical insights.

---

## 🏗️ System Architecture

```
                      ┌──────────────────────────────────────────┐
                      │    Goodbooks-10k Big Data Sources        │
                      │  • ratings.csv   (~6 Million rows)       │
                      │  • books.csv     (10,000 books)          │
                      │  • book_tags.csv (~1 Million tags)       │
                      │  • tags.csv      (Tag definitions)       │
                      └────────────────────┬─────────────────────┘
                                           │
                                           ▼
                      ┌──────────────────────────────────────────┐
                      │    Apache Spark (PySpark) Engine         │
                      │  1. Distributed CSV Parsing & Casting    │
                      │  2. Data Cleansing & Outlier Filtering   │
                      │  3. Per-Book Rating Count, Mean & StdDev │
                      │  4. User Activity Distribution Metrics   │
                      │  5. Tag Mining & Genre Extraction        │
                      │  6. Bayesian Weighted Rating (WR)        │
                      │  7. ALS Matrix Factorization Training    │
                      └────────────────────┬─────────────────────┘
                                           │
                                           ▼
                      ┌──────────────────────────────────────────┐
                      │           Processed Cache Layer          │
                      │  • data/processed/books_processed.csv    │
                      │  • data/processed/item_factors.csv       │
                      │  • data/processed/analytics_summary.json │
                      └────────────────────┬─────────────────────┘
                                           │
                                           ▼
                      ┌──────────────────────────────────────────┐
                      │        Recommendation Engine Layer       │
                      │  • Content-Based Filtering (TF-IDF)      │
                      │  • Collaborative Filtering (Spark ALS)   │
                      │  • Hybrid Blended Scoring                │
                      │  • Self-Exclusion Guarantee              │
                      └────────────────────┬─────────────────────┘
                                           │
                                           ▼
                      ┌──────────────────────────────────────────┐
                      │        Streamlit + Plotly Frontend       │
                      │  • Discover & Popular Books              │
                      │  • Personalized Recommender              │
                      │  • Search & Catalog Explorer             │
                      │  • Spark Big Data Analytics Dashboard    │
                      └──────────────────────────────────────────┘
```

---

## 🧠 Recommendation Algorithms Implemented

### 1. Popularity-Based Recommendations (Bayesian Weighted Rating)
To prevent books with a handful of 5-star reviews from outranking universally acclaimed books with millions of reviews, BookWise implements the Bayesian Weighted Rating formula (IMDB formula):

$$\text{WR} = \left(\frac{v}{v + m}\right) \cdot R + \left(\frac{m}{v + m}\right) \cdot C$$

- $v$ = Number of ratings for the book (`ratings_count`)
- $m$ = Minimum rating threshold (set to 500 ratings)
- $R$ = Average rating of the book
- $C$ = Mean rating across the entire catalog (~3.92)

### 2. Content-Based Filtering (TF-IDF + Cosine Similarity)
- Constructs a rich metadata soup combining book title, author (weighted x2), and top genre tags (weighted x2).
- Extracts uni-grams and bi-grams with sublinear term-frequency scaling (`TfidfVectorizer`).
- Computes pairwise cosine similarity vectors with linear kernel acceleration.
- Returns top $N$ most similar books with **strict self-exclusion** (a selected book is never recommended to itself).

### 3. Collaborative Filtering (PySpark ALS Latent Factors)
- Implements distributed **Alternating Least Squares (ALS)** matrix factorization using PySpark MLlib on all ~6 million interaction records.
- Learns 12 latent preference dimensions for each book, capturing hidden reader behavioral relationships (e.g. readers who loved *The Hunger Games* also highly rate *Catching Fire*, *Mockingjay*, and *Harry Potter*).
- Normalizes latent item vectors and computes vector cosine similarities.

### 4. Hybrid Blended Recommendation
- Fuses Content Similarity ($S_{content}$) with Collaborative Similarity ($S_{collab}$):
  $$\text{Score}_{hybrid} = w \cdot S_{content} + (1 - w) \cdot S_{collab}$$
- Balanced default ($w = 0.5$) provides both thematic relevance and community wisdom.

---

## 📊 Big Data Analytics Highlights (from PySpark Execution)
- **Total Processed Ratings:** 5,976,479 ratings
- **Total Unique Users:** 53,424 users
- **Total Cataloged Books:** 10,000 books
- **Global Mean Rating:** 3.92 / 5.00
- **Reader Engagement:** Mean of 111.87 ratings per user (long-tail power-law distribution)

---

## 📁 Project Structure

```
BDSMA/
├── app.py                     # Modern Streamlit web application & Plotly charts
├── recommendation.py          # Core recommendation engine (Content, ALS, Hybrid, Popularity)
├── data_processing.py         # PySpark data ingestion, aggregation, and ALS training pipeline
├── requirements.txt           # Python dependencies (pyspark, pandas, sklearn, streamlit, etc.)
├── launch.bat                 # One-click Windows launcher
├── test_engine.py             # Unit tests for recommendation algorithms
├── goodbooks-10k-master/      # Raw dataset files
│   ├── books.csv
│   ├── ratings.csv
│   ├── book_tags.csv
│   ├── tags.csv
│   └── to_read.csv
├── data/
│   ├── cache/                 # Temporary execution cache
│   └── processed/             # PySpark processed analytical datasets
│       ├── books_processed.csv
│       ├── item_factors.csv
│       └── analytics_summary.json
├── jdk17/                     # Portable OpenJDK 17 environment for PySpark
└── README.md                  # Project documentation
```

---

## 🚀 How to Run the Project

### Option 1: One-Click Launcher (Windows)
Double-click `launch.bat` in the project root directory. It automatically configures Java, loads the PySpark cache, and launches the Streamlit application.

### Option 2: Command Line (PowerShell)

1. Open PowerShell in the project directory `C:\Users\Asus\Desktop\BDSMA`:

```powershell
# Set Java and Python environment variables
$env:JAVA_HOME = "$PWD\jdk17"
$env:PATH = "$PWD\jdk17\bin;C:\Users\Asus\AppData\Local\Programs\Python\Python314;C:\Users\Asus\AppData\Local\Programs\Python\Python314\Scripts;" + $env:PATH
$env:PYSPARK_PYTHON = "C:\Users\Asus\AppData\Local\Programs\Python\Python314\python.exe"
$env:PYSPARK_DRIVER_PYTHON = "C:\Users\Asus\AppData\Local\Programs\Python\Python314\python.exe"
```

2. (Optional) Re-run the PySpark pipeline on the raw dataset:
```powershell
python data_processing.py --reprocess
```

3. Launch the Streamlit application:
```powershell
python -m streamlit run app.py
```

4. Open your browser and navigate to:
```
http://localhost:8501
```

---

## 🧪 Verification & Automated Tests
To run the automated recommendation verification suite:
```powershell
python test_engine.py
```

Output confirms:
- 10,000 books indexed
- Content-based recommendations verified (Mockingjay, Catching Fire, etc.)
- Collaborative ALS latent similarity verified
- Hybrid blending verified
- Self-recommendation avoidance verified (100% passed)
- Popularity Bayesian weighted rating verified

---

## 👥 Authors
Developed for the **College Big Data Analytics (BDSMA)** Microproject.

