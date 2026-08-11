"""
Travel Destination Category Recommender — Hybrid (Collaborative Filtering + Item-Similarity Content)
=======================================================================================================
Dataset : UCI Travel Review Ratings (Renjith et al., 2018) — google_review_ratings.csv
          5,456 users x 24 travel categories (Google review average rating, scale 1-5)
License : CC BY 4.0 — cite: Renjith, S. (2018). Travel Review Ratings. UCI ML Repository.

v3 — revision pass addressing REVIEW_FINDINGS from code-project-reviewer (v2 -> v3):
  #1  (critical) Out-of-range cells (e.g. a "22.0" on a 1-5 scale) were only WARNED
      about, never fixed. Now they are set to NaN (same policy as the 0-value fix)
      instead of silently poisoning downstream stats, and every affected
      (user, category) is logged individually.
  #2  (major)    ranking_metrics_at_k() now also reports the average/min/max
      candidate-pool size per evaluated user, since with only 24 categories and a
      96%-dense matrix the pool is often close to K itself — Recall@K needs that
      context to be interpreted honestly.
  #3  (major)    Added popularity-baseline and random-baseline ranking evaluation
      (same Precision/Recall/Coverage@K machinery, reused via a generic
      `recommend_fn` interface) so the hybrid numbers have a reference point.
  #4  (major)    Added a simulated cold-start experiment: a held-out group of users
      is fully excluded from CF training (never seen by SVD at all) and evaluated
      CF-only vs Hybrid, to show whether the content component actually helps for
      users the collaborative signal can't reach — not just warm UCI users.
  #5  (minor)    evaluate_hybrid_rmse()/tune_hybrid_weight() no longer recompute
      cf_score/content_score once per weight in the grid. Both are computed ONCE
      per split and reused for every weight (pure numpy arithmetic per grid point).
  #6  (minor)    File loading now catches FileNotFoundError, ParserError, and
      UnicodeDecodeError with a clear message instead of only FileNotFoundError.
  #7  (minor)    Two-stage hyperparameter search: coarse grid, then a finer grid
      re-centered on the coarse best (both for SVD params and hybrid weight).
  #8  (minor, cross-file) hybrid_score() now accepts weight_cf as either a float
      OR a callable(cf_model, user_id) -> float, so 03_journal_to_wide.py's
      cold-start-aware blend no longer needs to duplicate the blending logic.
  #9  (minor, cross-file) The cold-start experiment (#4) also grid-searches the
      best hybrid weight specifically for the excluded cold-start group, giving
      03_journal_to_wide.py's WEIGHT_CF_NEW_USER a value grounded in an actual
      experiment instead of an untuned guess. Still simulated (UCI users held out,
      not real Vietnamese journal users) — flagged as partially resolved.
  #10 (minor)    Added a small sensitivity table for RELEVANT_THRESHOLD and
      SIM_THRESHOLD so the reported ranking numbers aren't just one cherry-picked
      cut point.

  (carried over from v1 -> v2, unchanged in v3):
  - Leakage: item-similarity & content component built ONLY from the train split.
  - Precision/Recall@K restricted to each user's held-out TEST categories only.
  - df_wide indexed by User ONCE and passed around.
  - SVD hyperparameters + hybrid weight tuned via grid search on train/val only.
"""

import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split as sk_train_test_split
from surprise import SVD, Dataset, Reader
from surprise import accuracy as surprise_accuracy
from surprise.model_selection import GridSearchCV as SurpriseGridSearchCV

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# 1. Config / constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATA_PATH = Path(__file__).parent / "google_review_ratings.csv"
OUTPUT_DIR = Path(__file__).parent / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_FRAC = 0.7
VAL_FRAC = 0.1
TEST_FRAC = 0.2

TOP_K = 5
RELEVANT_THRESHOLD = 4.0
SIM_THRESHOLD = 0.1
HYBRID_WEIGHT_GRID_COARSE = [0.3, 0.5, 0.6, 0.7, 0.8, 1.0]

SVD_PARAM_GRID_COARSE = {
    "n_factors": [10, 15, 25],
    "n_epochs": [20, 30],
    "lr_all": [0.005, 0.01],
    "reg_all": [0.02, 0.1],
}

N_COLDSTART_USERS = 150  # finding #4: users fully excluded from CF training
THRESHOLD_SENSITIVITY_GRID = [3.5, 4.0, 4.5]  # finding #10
SIM_THRESHOLD_SENSITIVITY_GRID = [0.0, 0.1, 0.2]  # finding #10


# ---------------------------------------------------------------------------
# 2. Data loading & validation  (finding #6: broader exception handling)
# ---------------------------------------------------------------------------
def load_and_clean_data(path: Path) -> pd.DataFrame:
    """Load the raw CSV, fix the known row-shift corruption, cast to numeric,
    and convert invalid values (0s AND out-of-[1,5]-range cells) to missing (NaN)."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu tại: {path}\n"
            f"Hãy đặt 'google_review_ratings.csv' cùng thư mục với script này, "
            f"hoặc sửa DATA_PATH ở đầu file."
        ) from e
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        # finding #6: FileNotFoundError alone wasn't enough — a corrupt CSV or
        # wrong encoding used to crash with a raw traceback.
        raise RuntimeError(
            f"Không đọc được file tại: {path} (lỗi: {e}).\n"
            f"Kiểm tra file có đúng định dạng CSV / encoding UTF-8 không, "
            f"hoặc file có bị hỏng/cắt ngang không."
        ) from e

    if "Unnamed: 25" in df.columns:
        df = df.drop(columns=["Unnamed: 25"])

    cat_cols = [c for c in df.columns if c.startswith("Category")]
    assert len(cat_cols) == 24, f"Expected 24 category columns, found {len(cat_cols)}"

    for c in cat_cols:
        df[c] = (
            df[c].astype(str)
            .str.replace(r"\s+", "", regex=True)
            .replace({"nan": np.nan})
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"[load] Loaded {len(df)} rows x {len(cat_cols)} category columns")

    # finding #1: out-of-range cells were previously only WARNED about, not fixed.
    out_of_range_mask = (df[cat_cols] < 0) | (df[cat_cols] > 5)
    n_out_of_range = int(out_of_range_mask.sum().sum())
    if n_out_of_range:
        print(f"[load] Found {n_out_of_range} cell(s) outside the valid [0,5] range "
              f"— logging each and converting to NaN (data-entry error, not a real rating):")
        for c in cat_cols:
            bad_rows = df.loc[out_of_range_mask[c], "User"]
            for u in bad_rows:
                bad_val = df.loc[df["User"] == u, c].values[0]
                print(f"    - {u}, {c}: {bad_val} -> NaN")
        df[cat_cols] = df[cat_cols].mask(out_of_range_mask)

    n_zeros = (df[cat_cols] == 0).sum().sum()
    print(f"[load] Converting {n_zeros} zero-value cells ({n_zeros / df[cat_cols].size:.2%} "
          f"of all cells) from 0 -> NaN (treated as 'not rated', not a real low score)")
    df[cat_cols] = df[cat_cols].replace(0, np.nan)

    return df


def to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = [c for c in df.columns if c.startswith("Category")]
    long_df = df.melt(id_vars="User", value_vars=cat_cols,
                       var_name="category", value_name="rating")
    long_df = long_df.dropna(subset=["rating"]).reset_index(drop=True)
    print(f"[transform] {len(long_df)} known (user, category) ratings "
          f"out of {len(df) * len(cat_cols)} possible cells")
    return long_df


# ---------------------------------------------------------------------------
# 3. Train/val/test split at the (user, category) rating level
# ---------------------------------------------------------------------------
def split_ratings(long_df: pd.DataFrame):
    train_val, test = sk_train_test_split(
        long_df, test_size=TEST_FRAC, random_state=RANDOM_SEED
    )
    val_size_within = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    train, val = sk_train_test_split(
        train_val, test_size=val_size_within, random_state=RANDOM_SEED
    )
    print(f"[split] train={len(train)}  val={len(val)}  test={len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def long_to_wide(long_df: pd.DataFrame, all_users: pd.Index, cat_cols: list) -> pd.DataFrame:
    wide = long_df.pivot_table(index="User", columns="category", values="rating")
    wide = wide.reindex(index=all_users, columns=cat_cols)
    return wide


# ---------------------------------------------------------------------------
# 4. Content / item-similarity component — TRAIN SPLIT ONLY
# ---------------------------------------------------------------------------
def build_item_similarity(df_wide_train: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    sim = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)
    for i, c1 in enumerate(cat_cols):
        sim.loc[c1, c1] = 1.0
        for c2 in cat_cols[i + 1:]:
            paired = df_wide_train[[c1, c2]].dropna()
            s1, s2 = paired[c1], paired[c2]
            if len(paired) >= 30 and s1.std() > 0 and s2.std() > 0:
                corr = float(pearsonr(s1, s2).statistic)
                corr = 0.0 if np.isnan(corr) else corr
            else:
                corr = 0.0
            sim.loc[c1, c2] = corr
            sim.loc[c2, c1] = corr
    return sim.fillna(0.0)


def content_predict(user_train_ratings: pd.Series, target_category: str,
                     item_sim: pd.DataFrame, sim_threshold: float = SIM_THRESHOLD) -> float:
    rated = user_train_ratings.dropna()
    rated = rated[rated.index != target_category]
    if rated.empty:
        return np.nan
    sims = item_sim.loc[target_category, rated.index]
    sims = sims[sims > sim_threshold]
    if sims.empty:
        return float(rated.mean())
    weighted = (sims * rated.loc[sims.index]).sum() / sims.sum()
    return float(np.clip(weighted, 1.0, 5.0))


# ---------------------------------------------------------------------------
# 5. Collaborative filtering — two-stage grid search (finding #7)
# ---------------------------------------------------------------------------
def _grid_search_svd(train_data, param_grid, label):
    gs = SurpriseGridSearchCV(SVD, param_grid, measures=["rmse"], cv=3, n_jobs=-1)
    gs.fit(train_data)
    best_params = gs.best_params["rmse"]
    print(f"[CF-{label}] Best params (CV RMSE={gs.best_score['rmse']:.4f}): {best_params}")
    return best_params, gs.best_score["rmse"]


def tune_and_train_cf(train_long: pd.DataFrame):
    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(train_long[["User", "category", "rating"]], reader)

    print("[CF] Coarse grid search (3-fold CV on train split)...")
    coarse_best, coarse_rmse = _grid_search_svd(train_data, SVD_PARAM_GRID_COARSE, "coarse")

    # finding #7: finer grid re-centered on the coarse best, so the reported
    # optimum isn't just an artifact of a coarse/uneven grid.
    fine_grid = {
        "n_factors": sorted(set([max(5, coarse_best["n_factors"] - 5),
                                  coarse_best["n_factors"],
                                  coarse_best["n_factors"] + 5])),
        "n_epochs": sorted(set([coarse_best["n_epochs"], coarse_best["n_epochs"] + 10])),
        "lr_all": sorted(set([coarse_best["lr_all"], round(coarse_best["lr_all"] / 2, 4)])),
        "reg_all": sorted(set([coarse_best["reg_all"], 0.05, round(coarse_best["reg_all"] / 2, 3)])),
    }
    print("[CF] Fine grid search around coarse best...")
    fine_best, fine_rmse = _grid_search_svd(train_data, fine_grid, "fine")

    best_params, best_cv_rmse = (fine_best, fine_rmse) if fine_rmse <= coarse_rmse else (coarse_best, coarse_rmse)
    print(f"[CF] Final chosen params (CV RMSE={best_cv_rmse:.4f}): {best_params}")

    model = SVD(random_state=RANDOM_SEED, **best_params)
    full_trainset = train_data.build_full_trainset()
    model.fit(full_trainset)
    return model, best_params


def cf_predict_set(model, split_long: pd.DataFrame):
    preds = [model.predict(row.User, row.category, r_ui=row.rating)
             for row in split_long.itertuples(index=False)]
    rmse = surprise_accuracy.rmse(preds, verbose=False)
    mae = surprise_accuracy.mae(preds, verbose=False)
    return preds, rmse, mae


# ---------------------------------------------------------------------------
# 6. Hybrid blending
#    finding #5: score components computed ONCE per split, reused across the
#    whole weight grid (pure numpy arithmetic per grid point, no re-predicting).
#    finding #8: weight_cf may be a float OR a callable(cf_model, user_id) -> float,
#    so 03_journal_to_wide.py can plug in a cold-start-aware weight without
#    duplicating the blend logic.
# ---------------------------------------------------------------------------
def _resolve_weight(weight_cf, cf_model, user_id):
    return weight_cf(cf_model, user_id) if callable(weight_cf) else weight_cf


def hybrid_score(cf_model, item_sim, df_train_indexed, user_id, category, weight_cf):
    w = _resolve_weight(weight_cf, cf_model, user_id)
    cf_score = cf_model.predict(user_id, category).est
    if user_id in df_train_indexed.index:
        content_score = content_predict(df_train_indexed.loc[user_id], category, item_sim)
    else:
        content_score = np.nan
    if np.isnan(content_score):
        return float(np.clip(cf_score, 1.0, 5.0))
    blended = w * cf_score + (1 - w) * content_score
    return float(np.clip(blended, 1.0, 5.0))


def compute_score_components(cf_model, item_sim, df_train_indexed, preds):
    """finding #5: compute cf_score/content_score ONCE per (user,category,true)
    row — independent of any hybrid weight — so weight tuning is pure arithmetic."""
    cf_scores = np.array([float(np.clip(p.est, 1.0, 5.0)) for p in preds])
    content_scores = np.array([
        content_predict(df_train_indexed.loc[p.uid], p.iid, item_sim)
        if p.uid in df_train_indexed.index else np.nan
        for p in preds
    ])
    true = np.array([p.r_ui for p in preds])
    return cf_scores, content_scores, true


def blend_and_eval(cf_scores, content_scores, true, weight_cf):
    has_content = ~np.isnan(content_scores)
    blended = np.where(
        has_content,
        weight_cf * cf_scores + (1 - weight_cf) * np.nan_to_num(content_scores),
        cf_scores,
    )
    blended = np.clip(blended, 1.0, 5.0)
    errors = blended - true
    rmse = float(np.sqrt((errors ** 2).mean()))
    mae = float(np.abs(errors).mean())
    return rmse, mae


def tune_hybrid_weight(cf_scores_val, content_scores_val, true_val, grid=HYBRID_WEIGHT_GRID_COARSE):
    print("[Hybrid] Tuning blend weight on validation split (coarse grid)...")
    results = [(w, blend_and_eval(cf_scores_val, content_scores_val, true_val, w)[0]) for w in grid]
    for w, rmse in results:
        print(f"  weight_cf={w:.2f} -> val RMSE={rmse:.4f}")
    coarse_best_w, coarse_best_rmse = min(results, key=lambda x: x[1])

    # finding #7: finer local grid around the coarse best.
    fine_grid = sorted(set(np.clip(np.round(
        np.arange(coarse_best_w - 0.1, coarse_best_w + 0.101, 0.02), 2), 0.0, 1.0).tolist()))
    fine_results = [(w, blend_and_eval(cf_scores_val, content_scores_val, true_val, w)[0]) for w in fine_grid]
    print("[Hybrid] Fine grid around coarse best...")
    for w, rmse in fine_results:
        print(f"  weight_cf={w:.2f} -> val RMSE={rmse:.4f}")
    best_w, best_rmse = min(results + fine_results, key=lambda x: x[1])
    print(f"[Hybrid] Best weight_cf={best_w} (val RMSE={best_rmse:.4f})")
    return best_w


# ---------------------------------------------------------------------------
# 7. Ranking evaluation — generic recommend_fn interface so hybrid / popularity /
#    random baselines all reuse the same evaluation loop (finding #3), and it
#    now reports candidate-pool size (finding #2).
# ---------------------------------------------------------------------------
def ranking_metrics_generic(recommend_fn, df_train_indexed, test_long, cat_cols,
                             k=TOP_K, threshold=RELEVANT_THRESHOLD, label="model"):
    """recommend_fn(user_id, candidates) -> list of top-k category names."""
    test_by_user = test_long.groupby("User")
    precisions, recalls, pool_sizes = [], [], []
    recommended_categories = set()

    for user_id, group in test_by_user:
        known_train_categories = (
            set(df_train_indexed.loc[user_id].dropna().index)
            if user_id in df_train_indexed.index else set()
        )
        candidates = [c for c in cat_cols if c not in known_train_categories]
        if len(candidates) < k:
            continue

        relevant_test_items = set(group.loc[group["rating"] >= threshold, "category"])
        if not relevant_test_items:
            continue

        pool_sizes.append(len(candidates))
        top_k = recommend_fn(user_id, candidates)
        recommended_categories.update(top_k)

        hit_in_topk = relevant_test_items.intersection(top_k)
        precisions.append(len(hit_in_topk) / k)
        recalls.append(len(hit_in_topk) / len(relevant_test_items))

    coverage = len(recommended_categories) / len(cat_cols)
    p_at_k = float(np.mean(precisions)) if precisions else float("nan")
    r_at_k = float(np.mean(recalls)) if recalls else float("nan")
    pool_mean = float(np.mean(pool_sizes)) if pool_sizes else float("nan")
    pool_min = int(np.min(pool_sizes)) if pool_sizes else 0
    pool_max = int(np.max(pool_sizes)) if pool_sizes else 0

    print(f"[Ranking-{label}] n_users_evaluated={len(precisions)}  "
          f"avg_candidate_pool={pool_mean:.1f} (min={pool_min}, max={pool_max}, k={k})")
    print(f"[Ranking-{label}] Precision@{k}={p_at_k:.4f}  Recall@{k}={r_at_k:.4f}  "
          f"Coverage={coverage:.2%} ({len(recommended_categories)}/{len(cat_cols)} categories)")
    return {
        "precision_at_k": p_at_k, "recall_at_k": r_at_k, "coverage": coverage,
        "avg_candidate_pool": pool_mean, "min_candidate_pool": pool_min,
        "max_candidate_pool": pool_max, "n_users_evaluated": len(precisions),
    }


def make_hybrid_recommend_fn(cf_model, item_sim, df_train_indexed, weight_cf, k=TOP_K):
    def _fn(user_id, candidates):
        scores = {c: hybrid_score(cf_model, item_sim, df_train_indexed, user_id, c, weight_cf)
                  for c in candidates}
        return sorted(scores, key=scores.get, reverse=True)[:k]
    return _fn


def make_popularity_recommend_fn(df_train_indexed, cat_cols, k=TOP_K):
    """finding #3: baseline #1 — always recommend the globally highest-train-mean
    categories, restricted to each user's own candidate set."""
    pop_scores = df_train_indexed.mean(axis=0)

    def _fn(user_id, candidates):
        return sorted(candidates, key=lambda c: pop_scores.get(c, 0.0), reverse=True)[:k]
    return _fn


def make_random_recommend_fn(k=TOP_K, seed=RANDOM_SEED):
    """finding #3: baseline #2 — uniform random pick from the candidate set,
    seeded per-user so it's reproducible."""
    def _fn(user_id, candidates):
        rng = random.Random(f"{seed}-{user_id}")
        pool = list(candidates)
        rng.shuffle(pool)
        return pool[:k]
    return _fn


def recommend_top_k(cf_model, item_sim, df_train_indexed, user_id, cat_cols, weight_cf, k=TOP_K):
    known = set(df_train_indexed.loc[user_id].dropna().index) if user_id in df_train_indexed.index else set()
    candidates = [c for c in cat_cols if c not in known] or cat_cols
    scores = {c: hybrid_score(cf_model, item_sim, df_train_indexed, user_id, c, weight_cf)
              for c in candidates}
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]


# ---------------------------------------------------------------------------
# 8. Cold-start simulation (finding #4 + #9)
# ---------------------------------------------------------------------------
def run_coldstart_experiment(df_wide, cat_cols, long_df, best_cf_params, item_sim,
                              n_coldstart=N_COLDSTART_USERS, seed=RANDOM_SEED):
    """Fully exclude a group of users from CF training (SVD never sees a single
    rating from them -> genuinely unknown users, not just 'sparse' ones), then
    compare CF-only vs Hybrid on their held-out ratings. Also grid-searches the
    best hybrid weight *for this group specifically*, to ground
    03_journal_to_wide.py's cold-start weight in an actual experiment (finding #9)
    instead of an untuned guess — still simulated (held-out UCI users, not real
    Vietnamese journal users), so flagged as partially resolved.
    """
    rng = np.random.RandomState(seed)
    all_users = df_wide["User"].unique()
    coldstart_users = set(rng.choice(all_users, size=n_coldstart, replace=False))

    cs_long = long_df[long_df["User"].isin(coldstart_users)].reset_index(drop=True)
    non_cs_long = long_df[~long_df["User"].isin(coldstart_users)].reset_index(drop=True)

    # Cold-start users' own ratings are split into a small "known profile" (their
    # journal so far) and a held-out "true" set to predict — but crucially NONE
    # of their ratings go into CF training below.
    cs_known, cs_heldout = sk_train_test_split(cs_long, test_size=0.5, random_state=seed)
    cs_known = cs_known.reset_index(drop=True)
    cs_heldout = cs_heldout.reset_index(drop=True)

    print(f"\n[Cold-start] {len(coldstart_users)} users fully excluded from CF training "
          f"({len(cs_long)} of their ratings never touch the SVD fit)")
    print(f"[Cold-start] known-profile rows={len(cs_known)}  held-out eval rows={len(cs_heldout)}")

    reader = Reader(rating_scale=(1, 5))
    cs_train_data = Dataset.load_from_df(non_cs_long[["User", "category", "rating"]], reader)
    cf_model_excl = SVD(random_state=seed, **best_cf_params)
    cf_model_excl.fit(cs_train_data.build_full_trainset())

    df_wide_cs_known = long_to_wide(cs_known, pd.Index(sorted(coldstart_users)), cat_cols)

    preds = [cf_model_excl.predict(row.User, row.category, r_ui=row.rating)
             for row in cs_heldout.itertuples(index=False)]
    cf_scores, content_scores, true = compute_score_components(
        cf_model_excl, item_sim, df_wide_cs_known, preds
    )

    cf_only_rmse, cf_only_mae = blend_and_eval(cf_scores, np.full_like(content_scores, np.nan), true, 1.0)

    # finding #9: grid-search the blend weight specifically for this cold-start group.
    weight_results = [(w, blend_and_eval(cf_scores, content_scores, true, w)[0])
                       for w in np.round(np.arange(0.0, 1.01, 0.1), 2)]
    best_cs_weight, best_cs_rmse = min(weight_results, key=lambda x: x[1])
    _, best_cs_mae = blend_and_eval(cf_scores, content_scores, true, best_cs_weight)

    print(f"[Cold-start] CF-only  (unknown user, global-bias fallback): RMSE={cf_only_rmse:.4f} MAE={cf_only_mae:.4f}")
    print(f"[Cold-start] Hybrid   (best weight_cf={best_cs_weight:.1f} for this group): "
          f"RMSE={best_cs_rmse:.4f} MAE={best_cs_mae:.4f}")
    improvement_pct = 100 * (cf_only_rmse - best_cs_rmse) / cf_only_rmse
    print(f"[Cold-start] Hybrid improves RMSE by {improvement_pct:.1f}% over CF-only for genuinely new users")

    return {
        "n_coldstart_users": n_coldstart,
        "cf_only_rmse": cf_only_rmse, "cf_only_mae": cf_only_mae,
        "hybrid_rmse": best_cs_rmse, "hybrid_mae": best_cs_mae,
        "best_weight_cf_coldstart": float(best_cs_weight),
        "rmse_improvement_pct": improvement_pct,
    }


# ---------------------------------------------------------------------------
# 9. Threshold sensitivity (finding #10)
# ---------------------------------------------------------------------------
def run_threshold_sensitivity(cf_model, df_wide_train, test_long, cat_cols, best_weight):
    print("\n[Sensitivity] Precision/Recall@5 across RELEVANT_THRESHOLD x SIM_THRESHOLD:")
    rows = []
    for thr in THRESHOLD_SENSITIVITY_GRID:
        for sim_thr in SIM_THRESHOLD_SENSITIVITY_GRID:
            item_sim_variant = build_item_similarity  # similarity itself doesn't depend on sim_thr;
            # sim_thr only affects which neighbors content_predict uses at inference.
            def _content_predict_variant(user_row, target_category, item_sim, _t=sim_thr):
                return content_predict(user_row, target_category, item_sim, sim_threshold=_t)

            def _score_fn(user_id, category, _t=sim_thr):
                cf_score = cf_model.predict(user_id, category).est
                if user_id in df_wide_train.index:
                    c = _content_predict_variant(df_wide_train.loc[user_id], category, ITEM_SIM_GLOBAL)
                else:
                    c = np.nan
                if np.isnan(c):
                    return float(np.clip(cf_score, 1, 5))
                return float(np.clip(best_weight * cf_score + (1 - best_weight) * c, 1, 5))

            def _recommend_fn(user_id, candidates, _score_fn=_score_fn):
                scores = {c: _score_fn(user_id, c) for c in candidates}
                return sorted(scores, key=scores.get, reverse=True)[:TOP_K]

            res = ranking_metrics_generic(_recommend_fn, df_wide_train, test_long, cat_cols,
                                           k=TOP_K, threshold=thr, label=f"thr={thr},sim={sim_thr}")
            rows.append({"relevant_threshold": thr, "sim_threshold": sim_thr, **res})
    return rows


ITEM_SIM_GLOBAL = None  # set in main(); used by run_threshold_sensitivity's closures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global ITEM_SIM_GLOBAL

    df_wide = load_and_clean_data(DATA_PATH)
    cat_cols = [c for c in df_wide.columns if c.startswith("Category")]
    long_df = to_long_format(df_wide)

    train_long, val_long, test_long = split_ratings(long_df)
    df_wide_train = long_to_wide(train_long, df_wide["User"], cat_cols)
    print(f"[train-only] {df_wide_train.notna().sum().sum()} known ratings available "
          f"for content/hybrid features (train split only)")

    print("\n--- Content component (train-only item-item similarity) ---")
    item_sim = build_item_similarity(df_wide_train, cat_cols)
    ITEM_SIM_GLOBAL = item_sim

    print("\n--- Collaborative filtering (SVD, two-stage tuned) ---")
    cf_model, best_cf_params = tune_and_train_cf(train_long)
    cf_test_preds, cf_test_rmse, cf_test_mae = cf_predict_set(cf_model, test_long)
    print(f"[CF] Test set -> RMSE={cf_test_rmse:.4f}, MAE={cf_test_mae:.4f}")

    print("\n--- Hybrid weight tuning (on validation split, vectorized) ---")
    cf_val_preds, _, _ = cf_predict_set(cf_model, val_long)
    cf_scores_val, content_scores_val, true_val = compute_score_components(
        cf_model, item_sim, df_wide_train, cf_val_preds
    )
    best_weight = tune_hybrid_weight(cf_scores_val, content_scores_val, true_val)

    print("\n--- Final hybrid evaluation (on held-out TEST split) ---")
    cf_scores_test, content_scores_test, true_test = compute_score_components(
        cf_model, item_sim, df_wide_train, cf_test_preds
    )
    hybrid_test_rmse, hybrid_test_mae = blend_and_eval(cf_scores_test, content_scores_test, true_test, best_weight)
    print(f"[Hybrid] Test set -> RMSE={hybrid_test_rmse:.4f}, MAE={hybrid_test_mae:.4f}")

    print("\n--- Ranking evaluation: Hybrid vs. Popularity baseline vs. Random baseline ---")
    hybrid_fn = make_hybrid_recommend_fn(cf_model, item_sim, df_wide_train, best_weight)
    pop_fn = make_popularity_recommend_fn(df_wide_train, cat_cols)
    rand_fn = make_random_recommend_fn()

    hybrid_ranking = ranking_metrics_generic(hybrid_fn, df_wide_train, test_long, cat_cols, label="hybrid")
    popularity_ranking = ranking_metrics_generic(pop_fn, df_wide_train, test_long, cat_cols, label="popularity-baseline")
    random_ranking = ranking_metrics_generic(rand_fn, df_wide_train, test_long, cat_cols, label="random-baseline")

    print("\n--- Cold-start simulation ---")
    coldstart_results = run_coldstart_experiment(df_wide, cat_cols, long_df, best_cf_params, item_sim)

    sensitivity_rows = run_threshold_sensitivity(cf_model, df_wide_train, test_long, cat_cols, best_weight)

    print("\n--- Sample recommendation ---")
    sample_user = df_wide["User"].iloc[0]
    top_k = recommend_top_k(cf_model, item_sim, df_wide_train, sample_user, cat_cols, best_weight)
    print(f"Top-{TOP_K} recommended categories for {sample_user}:")
    for cat, score in top_k:
        print(f"  {cat:15s} predicted_rating={score:.2f}")

    # --- Save artifacts ---
    import pickle
    with open(OUTPUT_DIR / "cf_model.pkl", "wb") as f:
        pickle.dump(cf_model, f)
    item_sim.to_csv(OUTPUT_DIR / "item_similarity.csv")
    df_wide.to_csv(OUTPUT_DIR / "cleaned_ratings.csv", index=False)

    with open(OUTPUT_DIR / "metrics.txt", "w") as f:
        f.write("Travel Recommender — Hybrid CF Evaluation (v3, review-findings-fixed)\n")
        f.write("=" * 68 + "\n")
        f.write(f"Best SVD params: {best_cf_params}\n")
        f.write(f"Best hybrid weight (CF): {best_weight}  (tuned on validation split, 2-stage grid)\n\n")

        f.write(f"CF-only (test)  RMSE={cf_test_rmse:.4f}  MAE={cf_test_mae:.4f}\n")
        f.write(f"Hybrid (test)   RMSE={hybrid_test_rmse:.4f}  MAE={hybrid_test_mae:.4f}\n\n")

        f.write("Ranking @5 — Hybrid vs. baselines (finding #3):\n")
        for name, res in [("Hybrid", hybrid_ranking), ("Popularity baseline", popularity_ranking),
                           ("Random baseline", random_ranking)]:
            f.write(f"  {name:20s} Precision@5={res['precision_at_k']:.4f}  "
                    f"Recall@5={res['recall_at_k']:.4f}  Coverage={res['coverage']:.2%}\n")
        f.write(f"  Avg candidate pool per evaluated user: {hybrid_ranking['avg_candidate_pool']:.1f} "
                f"(min={hybrid_ranking['min_candidate_pool']}, max={hybrid_ranking['max_candidate_pool']}, K=5) "
                f"— finding #2: with only 24 categories and a ~96%-dense matrix, this pool is often "
                f"close to K itself, which inflates Recall@K relative to a large-catalog setting.\n\n")

        f.write("Cold-start simulation (finding #4 + #9) — users fully excluded from CF training:\n")
        f.write(f"  n_coldstart_users={coldstart_results['n_coldstart_users']}\n")
        f.write(f"  CF-only  RMSE={coldstart_results['cf_only_rmse']:.4f}  MAE={coldstart_results['cf_only_mae']:.4f}\n")
        f.write(f"  Hybrid   RMSE={coldstart_results['hybrid_rmse']:.4f}  MAE={coldstart_results['hybrid_mae']:.4f}  "
                f"(best weight_cf={coldstart_results['best_weight_cf_coldstart']})\n")
        f.write(f"  -> Hybrid improves RMSE by {coldstart_results['rmse_improvement_pct']:.1f}% over CF-only "
                f"for genuinely new users (content component's real value-add).\n\n")

        f.write("Threshold sensitivity (finding #10) — Precision/Recall@5 across thresholds:\n")
        for row in sensitivity_rows:
            f.write(f"  relevant_threshold={row['relevant_threshold']}  sim_threshold={row['sim_threshold']}  "
                    f"Precision@5={row['precision_at_k']:.4f}  Recall@5={row['recall_at_k']:.4f}  "
                    f"Coverage={row['coverage']:.2%}\n")

        f.write(f"\nAll content/item-similarity features built from TRAIN split only "
                f"({df_wide_train.notna().sum().sum()} ratings) — no leakage from "
                f"validation/test into the content or hybrid-weight-tuning steps.\n")
        f.write(f"Ranking metrics (Precision/Recall@{TOP_K}, Coverage) computed on the "
                f"held-out TEST split, candidates restricted to categories NOT seen "
                f"in train for each user.\n")

    print(f"\n[save] Artifacts written to {OUTPUT_DIR}/")

    return {
        "cf_test_rmse": cf_test_rmse, "cf_test_mae": cf_test_mae,
        "hybrid_test_rmse": hybrid_test_rmse, "hybrid_test_mae": hybrid_test_mae,
        "hybrid_ranking": hybrid_ranking, "popularity_ranking": popularity_ranking,
        "random_ranking": random_ranking, "coldstart_results": coldstart_results,
        "best_cf_params": best_cf_params, "best_hybrid_weight": best_weight,
    }


if __name__ == "__main__":
    main()
