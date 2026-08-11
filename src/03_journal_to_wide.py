"""
Journal -> Wide Matrix Bridge
=============================
Mảnh ghép nối "Bước E" trong sơ đồ đã bàn: gom journal/review thật (đã dịch category
qua category_crosswalk.csv) thành 1 bảng user x 24-UCI-category, ĐÚNG shape với
df_wide_train trong train_hybrid_recommender_v2.py — để tái sử dụng nguyên vẹn
cf_model, item_sim, hybrid_score(), content_predict(), recommend_top_k() đã có,
KHÔNG sửa 1 dòng nào trong file đó.

Input:
  - journal_log.csv       : user_id, google_category, rating (1-5), [date, place_name...]
  - category_crosswalk.csv: output của 02_category_crosswalk.py (google_category -> uci_category)

Output:
  - df_wide_journal: DataFrame index=user_id, columns=24 UCI category, giá trị=rating
    trung bình -> đút thẳng vào các hàm của v2.

LƯU Ý QUAN TRỌNG (cold-start CF): với user hoàn toàn mới (chưa từng có trong
train_long lúc fit SVD), cf_model.predict() sẽ trả về ước lượng mặc định (global
bias), KHÔNG cá nhân hoá được. Vì vậy user mới cần trọng số content cao hơn hẳn
0.8/0.2 mặc định — xem WEIGHT_CF_NEW_USER và hàm build_adaptive_weight_resolver()
cuối file.

v2 (bản này): sau review REVIEW_FINDINGS finding #8, bỏ hybrid_score_adaptive()
(từng duplicate gần như nguyên văn hybrid_score() của v2) — giờ tái dùng THẲNG
hybrid_score() từ train_hybrid_recommender_v3, vì hàm đó đã hỗ trợ weight_cf dạng
callable(cf_model, user_id) -> float, không chỉ float cố định. Điều này đảm bảo
2 nơi không bao giờ lệch logic blend với nhau nữa.

finding #9: WEIGHT_CF_NEW_USER trước đây là 0.3, tự ghi "chưa tune". Bản này cập
nhật thành 0.4 — lấy từ thử nghiệm cold-start MÔ PHỎNG trong v3
(run_coldstart_experiment: 150 user UCI bị loại HOÀN TOÀN khỏi tập train SVD, rồi
grid-search trọng số hybrid tốt nhất riêng cho nhóm đó). Đây vẫn là số liệu MÔ
PHỎNG (dùng user UCI ẩn đi, không phải user Việt Nam thật từ journal) — khi có
journal_log.csv thật, cần chạy lại quy trình tương tự trên chính nhóm user thật để
có con số đáng tin hơn.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config — chỉnh lại đường dẫn cho đúng máy bạn
# ---------------------------------------------------------------------------
JOURNAL_PATH = Path("journal_log.csv")
CROSSWALK_PATH = Path("artifacts/category_crosswalk.csv")
CLEANED_RATINGS_PATH = Path("cleaned_ratings.csv")  # output của v2, có cột "Category 1".."Category 24"

# THỨ TỰ NÀY QUAN TRỌNG: đây là thứ tự category thật ứng với "Category 1".."Category 24"
# trong dataset UCI Travel Review Ratings, dẫn từ tài liệu gốc (Renjith et al., 2018).
# Category 1 = churches, Category 2 = resorts, ... Category 24 = gardens.
# NẾU DANH SÁCH NÀY SAI THỨ TỰ, mọi điểm hybrid_score tính ra sẽ SAI Ở TẦNG Ý NGHĨA
# (silent — không crash, chỉ ra kết quả sai) vì cf_model/item_sim của v2 chỉ biết
# tên cột "Category N" chứ không biết "Category N" nghĩa là gì.
UCI_CATEGORIES = [
    "churches", "resorts", "beaches", "parks", "theatres", "museums", "malls",
    "zoo", "restaurants", "pubs/bars", "local services", "burger/pizza shops",
    "hotels/other lodgings", "juice bars", "art galleries", "dance clubs",
    "swimming pools", "gyms", "bakeries", "beauty & spas", "cafes",
    "view points", "monuments", "gardens",
]


def load_uci_category_order(cleaned_ratings_path: Path) -> list[str]:
    """Lấy đúng thứ tự tên cột 'Category 1'..'Category 24' từ file UCI gốc đã clean
    (output của v2), để build category_name_to_col khớp với cf_model/item_sim."""
    df = pd.read_csv(cleaned_ratings_path, nrows=1)
    cols = [c for c in df.columns if c.startswith("Category")]
    if len(cols) != len(UCI_CATEGORIES):
        raise AssertionError(
            f"Số cột Category trong {cleaned_ratings_path} ({len(cols)}) khác "
            f"len(UCI_CATEGORIES) ({len(UCI_CATEGORIES)}) — kiểm tra lại file/thứ tự trước khi tiếp tục."
        )
    return cols


def build_category_name_to_col(cleaned_ratings_path: Path) -> dict[str, str]:
    """Ghép UCI_CATEGORIES (tên thật, theo đúng thứ tự tài liệu Renjith 2018) với
    'Category N' (tên cột thật trong cf_model/item_sim) — đây là bước BẮT BUỘC
    trước khi gọi build_wide_from_journal(..., category_name_to_col=...), nếu
    không tên cột sẽ không khớp và hybrid_score_adaptive sẽ sai âm thầm."""
    cat_cols_order = load_uci_category_order(cleaned_ratings_path)
    mapping = dict(zip(UCI_CATEGORIES, cat_cols_order))
    print(f"[mapping] category_name_to_col: {mapping}")
    return mapping


def load_crosswalk(path: Path) -> dict[str, str]:
    """google_category (text tự do) -> uci_category (1 trong 24 nhãn chuẩn).
    Bỏ qua các dòng uci_category == 'unmapped'."""
    df = pd.read_csv(path)
    df = df[df["uci_category"] != "unmapped"]
    mapping = dict(zip(df["google_category"], df["uci_category"]))
    print(f"[crosswalk] Nạp {len(mapping)} ánh xạ google_category -> uci_category")
    return mapping


def load_journal(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"user_id", "google_category", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"journal_log.csv thiếu cột bắt buộc: {missing}")
    out_of_range = ((df["rating"] < 1) | (df["rating"] > 5)).sum()
    if out_of_range:
        print(f"[journal] CẢNH BÁO: {out_of_range} dòng rating ngoài khoảng [1,5], sẽ bị loại")
        df = df[(df["rating"] >= 1) & (df["rating"] <= 5)]
    print(f"[journal] Nạp {len(df)} dòng journal, {df['user_id'].nunique()} user")
    return df


def build_wide_from_journal(journal_df: pd.DataFrame, crosswalk: dict[str, str],
                             category_name_to_col: dict[str, str] | None = None) -> pd.DataFrame:
    """Cầu nối chính: journal (user, google_category, rating) -> wide matrix
    (user x 24 category), CÙNG SHAPE với df_wide_train trong v2.

    category_name_to_col: nếu truyền vào, đổi tên category thật (vd 'museums')
    thành tên cột kiểu UCI gốc (vd 'Category 6') để tương thích với cf_model đã
    train sẵn. Nếu None, giữ nguyên tên category thật (dùng khi bạn train lại
    CF model từ đầu trên dữ liệu journal, không cần khớp cf_model UCI cũ).
    """
    df = journal_df.copy()
    df["uci_category"] = df["google_category"].map(crosswalk)

    n_before = len(df)
    df = df.dropna(subset=["uci_category"])
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[wide] Bỏ {n_dropped} dòng journal có category không map được (unmapped) "
              f"— các dòng này KHÔNG dùng để tính điểm, không phải xoá nhầm.")

    # Cùng 1 user có thể ghé nhiều nơi cùng 1 uci_category -> lấy trung bình
    pivot = df.groupby(["user_id", "uci_category"])["rating"].mean().unstack("uci_category")
    pivot = pivot.reindex(columns=UCI_CATEGORIES)  # đảm bảo đủ 24 cột, thiếu -> NaN

    if category_name_to_col:
        pivot = pivot.rename(columns=category_name_to_col)

    n_known = pivot.notna().sum().sum()
    print(f"[wide] df_wide_journal: {pivot.shape[0]} user x {pivot.shape[1]} category, "
          f"{n_known} ô có rating ({n_known / pivot.size:.1%} lấp đầy)")
    return pivot


# ---------------------------------------------------------------------------
# Cold-start-aware blend — finding #8: KHÔNG còn duplicate hybrid_score() của v2/v3.
# Thay vào đó, build 1 "weight resolver" callable rồi truyền thẳng vào
# hybrid_score() gốc (hybrid_score() trong v3 nhận weight_cf là float HOẶC
# callable(cf_model, user_id) -> float).
# ---------------------------------------------------------------------------
# finding #9: 0.4 lấy từ thử nghiệm cold-start MÔ PHỎNG (xem docstring đầu file)
# — không phải số đoán tùy ý như bản trước (0.3), nhưng vẫn cần chạy lại trên
# journal thật khi có đủ dữ liệu để có con số đáng tin hơn.
WEIGHT_CF_KNOWN_USER = 0.88  # = best_weight từ train_hybrid_recommender_v3 (warm UCI users)
WEIGHT_CF_NEW_USER = 0.4     # = best_weight_cf_coldstart từ run_coldstart_experiment() trong v3


def build_adaptive_weight_resolver(weight_cf_known_user=WEIGHT_CF_KNOWN_USER,
                                    weight_cf_new_user=WEIGHT_CF_NEW_USER):
    """Trả về 1 callable(cf_model, user_id) -> float, đúng chữ ký mà
    hybrid_score() trong v3 chấp nhận cho tham số weight_cf — tự hạ trọng số CF
    cho user KHÔNG có trong trainset gốc (SVD cho user lạ chỉ trả về ước lượng
    mặc định, không phản ánh sở thích thật của họ), nên phần content (dựa trên
    journal thật) cần được tin tưởng nhiều hơn."""

    def _resolve(cf_model, user_id):
        is_known_user = (hasattr(cf_model, "trainset")
                          and cf_model.trainset.knows_user(user_id))
        return weight_cf_known_user if is_known_user else weight_cf_new_user

    return _resolve


def hybrid_score_for_journal_user(cf_model, item_sim, df_wide_journal, user_id, category):
    """Wrapper mỏng gọi thẳng hybrid_score() từ v3 với weight resolver cold-start-aware
    ở trên — không còn logic blend riêng ở đây, tránh 2 nơi lệch nhau (finding #8)."""
    from train_hybrid_recommender_v3 import hybrid_score  # tái dùng nguyên hàm v3
    weight_resolver = build_adaptive_weight_resolver()
    return hybrid_score(cf_model, item_sim, df_wide_journal, user_id, category, weight_resolver)


if __name__ == "__main__":
    crosswalk = load_crosswalk(CROSSWALK_PATH)
    journal = load_journal(JOURNAL_PATH)

    # BẮT BUỘC: build mapping tên thật -> "Category N" TRƯỚC khi đổi tên cột,
    # nếu không df_wide_journal sẽ mang tên cột không khớp cf_model/item_sim.
    category_name_to_col = build_category_name_to_col(CLEANED_RATINGS_PATH)
    df_wide_journal = build_wide_from_journal(journal, crosswalk,
                                               category_name_to_col=category_name_to_col)

    print("\n--- df_wide_journal (preview, cột đã đổi tên khớp cf_model/item_sim) ---")
    print(df_wide_journal.head())

    print("\nBước tiếp theo (cần cf_model.pkl + item_similarity.csv từ v3 đã train):")
    print("  from train_hybrid_recommender_v3 import recommend_top_k")
    print("  weight_resolver = build_adaptive_weight_resolver()")
    print("  recommend_top_k(cf_model, item_sim, df_wide_journal, user_id, cat_cols, weight_cf=weight_resolver)")
    print("  # weight_resolver tự hạ xuống 0.4 cho user không có trong train gốc "
          "(số này đến từ thử nghiệm cold-start mô phỏng trong v3, xem docstring đầu file)")
