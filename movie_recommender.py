"""
============================================================
  MOVIE RECOMMENDATION SYSTEM — Linear Algebra Approach
  Dataset: MovieLens 100K (https://grouplens.org/datasets/movielens/100k/)
  Version : 3.0  —  Interactive CLI Edition
============================================================

Architecture Overview
---------------------
1.  Data ingestion & cleaning             (pandas)
2.  User-Movie rating matrix              (numpy)
3.  Gaussian elimination / LU             (scipy)
4.  Content feature matrix                (TF-IDF / one-hot)
5.  Vector-space user profiles            (numpy dot products)
6.  Gram-Schmidt orthogonalisation        (numpy)
7.  Least-squares rating prediction       (numpy.linalg.lstsq)
8.  Eigenvalue / SVD trend analysis       (numpy.linalg)
9.  Collaborative + content hybrid score  (weighted blend)
10. Interactive Top-N recommendation CLI
"""

# -----------------------------------------------------------
# 0. IMPORTS
# -----------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.linalg import lu
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings, textwrap, sys

warnings.filterwarnings("ignore")
np.random.seed(42)

# ANSI colours for a nicer terminal experience
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"
LINE   = "=" * 64


# -----------------------------------------------------------
# 1. CONFIGURATION  (defaults — all overridable at runtime)
# -----------------------------------------------------------
CONFIG = {
    "ratings_path"  : "ml-100k/u.data",
    "movies_path"   : "ml-100k/u.item",
    "min_ratings"   : 20,
    "latent_factors": 20,
    "top_n"         : 10,
    "cf_weight"     : 0.60,
    "cb_weight"     : 0.40,
    "fill_value"    : 0.0,
}

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
    # synthetic-data genres (present when real data not available)
    "Thriller",
]


# -----------------------------------------------------------
# 2. DATA LOADING
# -----------------------------------------------------------

def load_movielens(cfg: dict):
    try:
        ratings = pd.read_csv(
            cfg["ratings_path"], sep="\t",
            names=["user_id", "movie_id", "rating", "timestamp"]
        )
        genre_cols = [
            "Action", "Adventure", "Animation", "Children", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
            "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
            "Thriller", "War", "Western"
        ]
        movies = pd.read_csv(
            cfg["movies_path"], sep="|", encoding="latin-1",
            names=["movie_id","title","release_date","video_release","imdb_url",
                   "unknown"] + genre_cols
        )
        movies["genres"] = movies[genre_cols].apply(
            lambda r: [g for g in genre_cols if r[g] == 1], axis=1
        )
        print(f"  {GREEN}Real MovieLens data loaded.{RESET}")
    except FileNotFoundError:
        print(f"  {YELLOW}MovieLens files not found — generating synthetic dataset.{RESET}")
        ratings, movies = _synthetic_data()
    return ratings, movies


def _synthetic_data(n_users: int = 200, n_movies: int = 100):
    genres_pool = [
        "Action", "Comedy", "Drama", "Sci-Fi", "Romance",
        "Thriller", "Horror", "Animation", "Documentary"
    ]
    movies = pd.DataFrame({
        "movie_id": range(1, n_movies + 1),
        "title"   : [f"Movie_{i}" for i in range(1, n_movies + 1)],
        "genres"  : [
            list(np.random.choice(genres_pool,
                 size=np.random.randint(1, 4), replace=False))
            for _ in range(n_movies)
        ],
        "year": np.random.randint(1980, 2024, n_movies),
    })
    rows = []
    for u in range(1, n_users + 1):
        sampled = np.random.choice(n_movies,
                                   size=max(5, int(n_movies * 0.15)),
                                   replace=False)
        for m in sampled:
            rows.append({"user_id": u, "movie_id": int(m)+1,
                          "rating": float(np.random.randint(1, 6))})
    return pd.DataFrame(rows), movies


# -----------------------------------------------------------
# 3. RATING MATRIX
# -----------------------------------------------------------

def build_rating_matrix(ratings: pd.DataFrame, fill: float = 0.0) -> pd.DataFrame:
    R = ratings.pivot_table(index="user_id", columns="movie_id",
                            values="rating", aggfunc="mean").fillna(fill)
    sparsity = 100.0 * (R == fill).values.mean()
    print(f"\n[Rating Matrix]  shape={R.shape}  sparsity={sparsity:.1f}%")
    return R


# -----------------------------------------------------------
# 4. GAUSSIAN ELIMINATION / LU
# -----------------------------------------------------------

def gaussian_elimination_analysis(R: np.ndarray) -> dict:
    sample   = R[:min(R.shape[0], 200), :min(R.shape[1], 200)]
    P, L, U  = lu(sample)
    rank     = int(np.sum(np.abs(np.diag(U)) > 1e-6))
    null_dim = sample.shape[1] - rank
    print(f"\n[Gaussian Elimination / LU]")
    print(f"    Sub-matrix analysed  : {sample.shape}")
    print(f"    Numerical rank       : {rank}  (independent taste dimensions)")
    print(f"    Null-space dimension : {null_dim}")
    return {"P": P, "L": L, "U": U, "rank": rank, "null_dim": null_dim}


# -----------------------------------------------------------
# 5. USER PREFERENCE PROFILES
# -----------------------------------------------------------

def build_user_profiles(R_df, movies_df, genre_cols) -> pd.DataFrame:
    mlb = MultiLabelBinarizer(classes=genre_cols)
    genre_matrix = pd.DataFrame(
        mlb.fit_transform(movies_df["genres"]),
        index=movies_df["movie_id"], columns=genre_cols
    )
    profiles = {}
    for uid in R_df.index:
        mask    = R_df.loc[uid] > 0
        movies_ = R_df.columns[mask]
        rvec    = R_df.loc[uid, movies_].values
        if len(rvec) == 0:
            profiles[uid] = np.zeros(len(genre_cols))
            continue
        gm = genre_matrix.reindex(movies_).fillna(0).values
        profiles[uid] = (gm.T * rvec).T.sum(axis=0) / (rvec.sum() + 1e-9)
    profile_df = pd.DataFrame(profiles, index=genre_cols).T
    print(f"\n[User Preference Profiles]  shape={profile_df.shape}")
    return profile_df


# -----------------------------------------------------------
# 6. GRAM-SCHMIDT ORTHOGONALISATION
# -----------------------------------------------------------

def gram_schmidt(A: np.ndarray) -> np.ndarray:
    Q = np.zeros_like(A, dtype=float)
    for i in range(len(A)):
        u = A[i].copy().astype(float)
        for j in range(i):
            u -= np.dot(Q[j], u) * Q[j]
        n = np.linalg.norm(u)
        Q[i] = u / n if n > 1e-10 else u
    return Q


def orthogonalise_taste_factors(profiles_df: pd.DataFrame,
                                n_factors: int = 5) -> np.ndarray:
    norms   = np.linalg.norm(profiles_df.values, axis=1)
    top_idx = np.argsort(norms)[::-1][:n_factors]
    Q       = gram_schmidt(profiles_df.values[top_idx])
    err     = np.abs(Q @ Q.T - np.eye(n_factors)).max()
    print(f"\n[Gram-Schmidt Orthogonalisation]")
    print(f"    Taste factors built  : {n_factors}")
    print(f"    Orthogonality error  : {err:.2e}  (< 1e-10 is perfect)")
    return Q


# -----------------------------------------------------------
# 7. LEAST-SQUARES PREDICTION  (silent flag for MAE loop)
# -----------------------------------------------------------

def least_squares_predict(R_df, user_id: int,
                          n_factors: int = 10,
                          silent: bool = False) -> pd.Series:
    R = R_df.values.astype(float)
    _, sigma, Vt = np.linalg.svd(R, full_matrices=False)
    k    = min(n_factors, len(sigma))
    X    = Vt[:k].T

    row  = R_df.loc[user_id].values
    mask = row > 0
    y    = row[mask];  Xk = X[mask]

    if len(y) < 2:
        if not silent:
            print(f"    User {user_id} has too few ratings; using mean fill.")
        return pd.Series(R_df.mean(), index=R_df.columns)

    beta, res, _, _ = np.linalg.lstsq(Xk, y, rcond=None)
    if not silent:
        print(f"\n[Least-Squares Prediction]  user={user_id}  k={k}")
        if len(res):
            print(f"    Residual MSE  : {float(res[0])/max(len(y),1):.4f}")
    return pd.Series(X @ beta, index=R_df.columns)


# -----------------------------------------------------------
# 8. SVD TREND ANALYSIS
# -----------------------------------------------------------

def svd_trend_analysis(R_df: pd.DataFrame, n_factors: int = 20) -> dict:
    R          = R_df.values.astype(float)
    U, sigma, Vt = np.linalg.svd(R, full_matrices=False)
    k          = min(n_factors, len(sigma))
    explained  = np.cumsum(sigma[:k]**2) / np.sum(sigma**2) * 100
    print(f"\n[SVD / Eigenvalue Trend Analysis]")
    print(f"    Top-{k} singular values : {np.round(sigma[:k], 2)}")
    print(f"    Variance explained    : {explained[-1]:.1f}%  (top {k} factors)")
    return {"U": U[:,:k]*sigma[:k], "Vt": Vt, "sigma": sigma,
            "explained": explained, "movie_lat": Vt[:k].T}


# -----------------------------------------------------------
# 9. CONTENT SIMILARITY
# -----------------------------------------------------------

def content_similarity_matrix(movies_df: pd.DataFrame) -> np.ndarray:
    gs   = movies_df["genres"].apply(
        lambda g: " ".join(g) if isinstance(g, list) else str(g))
    tfidf = TfidfVectorizer(token_pattern=r"[A-Za-z\-]+")
    sim   = cosine_similarity(tfidf.fit_transform(gs))
    print(f"\n[Content Similarity Matrix]  shape={sim.shape}")
    return sim


def content_based_score(user_id, R_df, movies_df,
                        sim_matrix, preferred_genres) -> pd.Series:
    movie_ids = movies_df["movie_id"].values
    n         = len(movie_ids)

    if user_id in R_df.index:
        ur        = R_df.loc[user_id]
        liked_ids = ur[ur >= 4].index.tolist()
        liked_idx = [int(np.where(movie_ids == m)[0][0])
                     for m in liked_ids if m in movie_ids]
    else:
        liked_idx = []

    liked_sim = (sim_matrix[:, liked_idx].mean(axis=1)
                 if liked_idx else np.zeros(n))

    mlb       = MultiLabelBinarizer()
    gmat      = mlb.fit_transform(movies_df["genres"].tolist())
    pref      = np.zeros(gmat.shape[1])
    for g in preferred_genres:
        if g in mlb.classes_:
            pref[list(mlb.classes_).index(g)] = 1.0
    genre_sim = gmat @ pref / (np.linalg.norm(pref) + 1e-9)

    sc       = MinMaxScaler()
    combined = (0.6 * sc.fit_transform(liked_sim.reshape(-1,1)).flatten()
              + 0.4 * sc.fit_transform(genre_sim.reshape(-1,1)).flatten())
    return pd.Series(combined, index=movie_ids)


# -----------------------------------------------------------
# 10. COLLABORATIVE SCORE
# -----------------------------------------------------------

def collaborative_score(user_id, R_df, svd_result) -> pd.Series:
    U_lat = svd_result["U"];  Vt = svd_result["Vt"];  k = U_lat.shape[1]
    if user_id not in R_df.index:
        return pd.Series(np.zeros(R_df.shape[1]), index=R_df.columns)
    u_idx = list(R_df.index).index(user_id)
    return pd.Series(U_lat[u_idx] @ Vt[:k], index=R_df.columns)


# -----------------------------------------------------------
# 11. HYBRID RECOMMENDER
# -----------------------------------------------------------

def hybrid_recommend(user_id, R_df, movies_df,
                     preferred_genres, cfg) -> pd.DataFrame:
    svd       = svd_trend_analysis(R_df, n_factors=cfg["latent_factors"])
    cf_scores = collaborative_score(user_id, R_df, svd)
    sim_mat   = content_similarity_matrix(movies_df)
    cb_scores = content_based_score(user_id, R_df, movies_df,
                                    sim_mat, preferred_genres)
    ls_pred   = least_squares_predict(R_df, user_id,
                                      n_factors=cfg["latent_factors"])

    all_ids = movies_df["movie_id"].values

    def align(s):
        return s.reindex(all_ids).fillna(0.0).values
    def norm01(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo + 1e-9)

    hybrid = (cfg["cf_weight"] * norm01(align(cf_scores))
            + cfg["cb_weight"] * norm01(align(cb_scores))
            + 0.10             * norm01(align(ls_pred)))

    rated = (set(R_df.columns[R_df.loc[user_id] > 0])
             if user_id in R_df.index else set())

    out = movies_df.copy()
    out["hybrid_score"] = hybrid
    out = out[~out["movie_id"].isin(rated)]
    out = out.sort_values("hybrid_score", ascending=False)
    top = out.head(cfg["top_n"])[
        ["movie_id","title","genres","hybrid_score"]
    ].reset_index(drop=True)
    top.index += 1
    return top


# -----------------------------------------------------------
# 12. EVALUATION  (silent inner loop → no per-user spam)
# -----------------------------------------------------------

def evaluate_mae(R_df: pd.DataFrame, cfg: dict,
                 sample_users: int = 20) -> float:
    errors     = []
    candidates = [u for u in R_df.index if (R_df.loc[u] > 0).sum() >= 5]
    test_users = candidates[:sample_users]

    for u in test_users:
        cols     = R_df.columns[R_df.loc[u] > 0].tolist()
        hold     = np.random.choice(cols)
        actual   = R_df.loc[u, hold]
        Rm       = R_df.copy(); Rm.loc[u, hold] = 0.0
        # silent=True suppresses the per-user print lines
        pred     = least_squares_predict(Rm, u, cfg["latent_factors"],
                                         silent=True)
        errors.append(abs(actual - pred.get(hold, 0.0)))

    mae = float(np.mean(errors))
    print(f"\n[Evaluation]  Leave-one-out MAE"
          f" (n={sample_users} users) : {mae:.4f}")
    return mae


# -----------------------------------------------------------
# 13. INTERACTIVE USER INPUT HELPERS
# -----------------------------------------------------------

def _prompt(msg: str, default=None) -> str:
    """Print a coloured prompt and return stripped input."""
    suffix = f" [{default}]" if default is not None else ""
    try:
        val = input(f"{CYAN}{msg}{suffix}: {RESET}").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return val if val else (str(default) if default is not None else "")


def get_user_id(valid_ids: list) -> int:
    """
    Ask for a user ID.
    If the user presses Enter with no input, or types 0,
    we treat it as a 'new user' (ID = -1).
    """
    id_range = f"1–{max(valid_ids)}" if valid_ids else "any"
    print(f"\n{DIM}  Known user IDs in this dataset: {id_range}{RESET}")
    print(f"{DIM}  Press Enter (or type 0) to proceed as a NEW user.{RESET}")
    raw = _prompt("  Enter your user ID", default=0)
    try:
        uid = int(raw)
    except ValueError:
        uid = 0
    if uid <= 0 or uid not in valid_ids:
        if uid > 0:
            print(f"  {YELLOW}User {uid} not found — treating as new user.{RESET}")
        return -1
    return uid


def get_genres(available: list) -> list:
    """
    Show a numbered genre menu and let the user pick by number or name.
    """
    print(f"\n{DIM}  Available genres:{RESET}")
    for i, g in enumerate(available, 1):
        print(f"    {i:2}. {g}")
    print(f"{DIM}  Enter numbers (e.g. 1,4,6) OR genre names (e.g. Action,Sci-Fi).{RESET}")
    raw = _prompt("  Your preferred genres", default="Action,Sci-Fi,Thriller")

    chosen = []
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(available):
                chosen.append(available[idx])
        else:
            # fuzzy: accept partial / case-insensitive match
            matches = [g for g in available
                       if token.lower() in g.lower()]
            chosen.extend(matches)

    chosen = list(dict.fromkeys(chosen))  # deduplicate, preserve order
    if not chosen:
        print(f"  {YELLOW}No valid genres selected — defaulting to Action, Sci-Fi.{RESET}")
        chosen = ["Action", "Sci-Fi"]
    return chosen


def get_top_n() -> int:
    raw = _prompt("  How many recommendations?", default=10)
    try:
        n = int(raw)
        return max(1, min(n, 50))
    except ValueError:
        return 10


def get_weights() -> tuple[float, float]:
    print(f"\n{DIM}  Hybrid blend (collaborative vs content-based).{RESET}")
    print(f"{DIM}  Collaborative: SVD user-similarity  |  Content: genre/TF-IDF match.{RESET}")
    raw = _prompt("  Collaborative weight (0–100%)", default=60)
    try:
        cf = max(0, min(int(raw), 100)) / 100.0
    except ValueError:
        cf = 0.60
    cb = round(1.0 - cf - 0.10, 2)          # 10% always reserved for lstsq
    cb = max(0.0, cb)
    print(f"  → CF={cf:.0%}  CB={cb:.0%}  Least-squares correction=10%")
    return cf, cb


def show_user_history(user_id: int, R_df: pd.DataFrame,
                      movies_df: pd.DataFrame, n: int = 5):
    """Print a short summary of what the user has already rated."""
    if user_id not in R_df.index:
        return
    rated = R_df.loc[user_id]
    rated = rated[rated > 0].sort_values(ascending=False).head(n)
    if rated.empty:
        return
    print(f"\n{DIM}  Top-rated movies by user {user_id} (history snapshot):{RESET}")
    for mid, score in rated.items():
        row = movies_df[movies_df["movie_id"] == mid]
        title = row["title"].values[0] if not row.empty else f"Movie_{mid}"
        genres = row["genres"].values[0] if not row.empty else []
        g = ", ".join(genres) if isinstance(genres, list) else genres
        print(f"    ★ {score:.0f}  {title}  [{g}]")


# -----------------------------------------------------------
# 14. PRINT RECOMMENDATIONS
# -----------------------------------------------------------

def print_recommendations(recs: pd.DataFrame, user_id: int,
                           genres: list, cfg: dict):
    label = f"USER {user_id}" if user_id > 0 else "NEW USER"
    pref  = ", ".join(genres)
    print(f"\n{BOLD}{LINE}{RESET}")
    print(f"{BOLD}  TOP {cfg['top_n']} RECOMMENDATIONS FOR {label}{RESET}")
    print(f"{DIM}  Preferred genres : {pref}{RESET}")
    print(f"{BOLD}{LINE}{RESET}")
    for rank, row in recs.iterrows():
        g = (", ".join(row["genres"])
             if isinstance(row["genres"], list) else row["genres"])
        score_bar = "█" * int(row["hybrid_score"] * 20)
        print(f"  {BOLD}{rank:2}.{RESET} {GREEN}{row['title']:<35}{RESET}"
              f"  {DIM}[{g}]{RESET}"
              f"  {YELLOW}{score_bar:<20}{RESET} {row['hybrid_score']:.3f}")
    print(f"{BOLD}{LINE}{RESET}")


# -----------------------------------------------------------
# 15. MAIN
# -----------------------------------------------------------

def main():
    print(f"\n{BOLD}{LINE}")
    print("  MOVIE RECOMMENDATION SYSTEM  —  Linear Algebra Edition v3.0")
    print(f"{LINE}{RESET}")

    cfg = CONFIG.copy()

    # ── Load & filter data ──────────────────────────────────────
    ratings_df, movies_df = load_movielens(cfg)
    genre_cols = sorted({g for gs in movies_df["genres"] for g in gs})

    counts     = ratings_df["movie_id"].value_counts()
    valid      = counts[counts >= cfg["min_ratings"]].index
    ratings_df = ratings_df[ratings_df["movie_id"].isin(valid)]
    movies_df  = movies_df[movies_df["movie_id"].isin(valid)].reset_index(drop=True)
    print(f"\n[Filter]  {len(movies_df)} movies kept "
          f"(each >= {cfg['min_ratings']} ratings)")

    # ── Build pipeline once ─────────────────────────────────────
    R_df        = build_rating_matrix(ratings_df, fill=cfg["fill_value"])
    gauss_result = gaussian_elimination_analysis(R_df.values)
    profiles_df  = build_user_profiles(R_df, movies_df, genre_cols)
    _            = orthogonalise_taste_factors(profiles_df,
                       n_factors=min(5, len(genre_cols)))
    mae          = evaluate_mae(R_df, cfg,
                       sample_users=min(20, len(R_df)))

    valid_ids = sorted(R_df.index.tolist())

    # ── Interactive recommendation loop ────────────────────────
    while True:
        print(f"\n{BOLD}{'─'*64}{RESET}")
        print(f"{BOLD}  NEW RECOMMENDATION SESSION{RESET}")
        print(f"{'─'*64}")

        # 1. User ID
        uid = get_user_id(valid_ids)

        # 2. Show what this user has rated (context)
        show_user_history(uid, R_df, movies_df)

        # 3. Genre preferences
        genres = get_genres(genre_cols)

        # 4. How many recommendations
        cfg["top_n"] = get_top_n()

        # 5. Hybrid weights
        cf_w, cb_w          = get_weights()
        cfg["cf_weight"]    = cf_w
        cfg["cb_weight"]    = cb_w

        # 6. Run recommender
        print(f"\n{DIM}  Computing recommendations …{RESET}")
        if uid == -1:
            # New user: add a temporary row of zeros so the pipeline works
            dummy_row        = pd.DataFrame(
                np.zeros((1, R_df.shape[1])),
                index=[-1], columns=R_df.columns
            )
            R_with_new       = pd.concat([R_df, dummy_row])
            recs = hybrid_recommend(-1, R_with_new, movies_df, genres, cfg)
        else:
            recs = hybrid_recommend(uid, R_df, movies_df, genres, cfg)

        print_recommendations(recs, uid, genres, cfg)

        # 7. Model quality reminder
        print(f"{DIM}  Model MAE (leave-one-out, scale 1–5): {mae:.4f}{RESET}")

        # 8. Go again?
        again = _prompt("\n  Get recommendations for another user? (y/n)",
                        default="y").lower()
        if again not in ("y", "yes", ""):
            break

    print(f"\n{GREEN}  Done. Enjoy your movies!{RESET}\n")


if __name__ == "__main__":
    main()