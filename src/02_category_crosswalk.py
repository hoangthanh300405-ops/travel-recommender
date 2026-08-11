"""
Category Crosswalk — UCI Travel Review Ratings (24 categories) <-> Google Local Review Data (US)
====================================================================================================
Mục đích: hai dataset không có user_id/place_id chung, nên không thể join trực tiếp.
Script này xây bảng ánh xạ Ở TẦNG CATEGORY: mỗi category text tự do của Google Local
(vd "Pizza restaurant", "Art museum") được gán vào 1 trong 24 category chuẩn của UCI,
để sau này dùng lại item_similarity.csv / hybrid CF đã train trên UCI làm "prior" cho
các địa điểm thật lấy từ Google Local.

Input : file metadata đã tải thủ công từ
        https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/
        (vd meta-Vermont.json.gz) — KHÔNG tự động tải trong script này vì domain đó
        không nằm trong whitelist mạng của môi trường chạy script.
Output: category_crosswalk.csv — mỗi dòng là 1 google category text gặp trong dữ liệu,
        kèm UCI category được gán, phương pháp gán (keyword / embedding / unmapped),
        và độ phổ biến (số business có category đó).

Chiến lược 2 tầng:
  1. Keyword rule (ưu tiên, minh bạch, dễ audit) — match substring không phân biệt hoa/thường.
  2. Fallback bằng sentence-transformers cosine similarity cho các category không khớp
     keyword nào — để không phải bỏ sót, nhưng luôn gắn nhãn "embedding" để biết đây là
     suy đoán, cần review thủ công trước khi tin tưởng hoàn toàn.
  Category không đạt ngưỡng similarity nào -> "unmapped", giữ nguyên để báo cáo minh bạch
  (đúng tinh thần đã nêu: "local services", "juice bars" khó có tương đương rõ ràng).
"""

import gzip
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd


def normalize_text(text: str) -> str:
    """Hạ chữ thường + bỏ dấu Unicode (vd 'Café' -> 'cafe') để so khớp keyword
    ổn định, kể cả khi sau này mở rộng sang category text tiếng Việt có dấu."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
# Đường dẫn tới file metadata bạn đã tải thủ công (đổi lại cho đúng máy bạn)
META_PATH = Path("meta-Vermont.json.gz")
OUTPUT_DIR = Path(__file__).parent / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_SIM_THRESHOLD = 0.45  # dưới ngưỡng này -> để "unmapped" thay vì đoán bừa

# 24 category chuẩn của UCI (Renjith et al., 2018)
UCI_CATEGORIES = [
    "churches", "resorts", "beaches", "parks", "theatres", "museums", "malls",
    "zoo", "restaurants", "pubs/bars", "local services", "burger/pizza shops",
    "hotels/other lodgings", "juice bars", "art galleries", "dance clubs",
    "swimming pools", "gyms", "bakeries", "beauty & spas", "cafes",
    "view points", "monuments", "gardens",
]

# Keyword rules: match substring (lowercase) trong google category text.
# THỨ TỰ QUAN TRỌNG: dict được duyệt theo thứ tự khai báo, category CỤ THỂ hơn
# phải đứng TRƯỚC category tổng quát hơn (vd "burger/pizza shops" trước "restaurants",
# nếu không "Pizza restaurant" sẽ bị khớp nhầm vào "restaurants" trước khi tới "pizza").
# "local services" và "juice bars" cố tình để keyword hẹp/rỗng — đúng như đã
# cảnh báo trước đó, hai category này khó có tương đương rõ ràng ở taxonomy khác.
KEYWORD_MAP = {
    "burger/pizza shops": ["pizza", "burger", "fast food"],
    "juice bars": ["juice bar", "juice shop", "smoothie"],
    "art galleries": ["art gallery", "gallery"],
    "churches": ["church", "chapel", "cathedral", "place of worship", "religious"],
    "resorts": ["resort"],
    "beaches": ["beach"],
    "parks": ["park", "picnic area", "recreation area", "nature preserve"],
    "theatres": ["theatre", "theater", "cinema", "movie theater", "performing arts"],
    "museums": ["museum"],
    "malls": ["mall", "shopping center", "shopping mall", "outlet store"],
    "zoo": ["zoo", "aquarium", "wildlife park"],
    "pubs/bars": ["bar", "pub", "tavern", "brewery", "wine bar", "cocktail"],
    "local services": [],  # không có keyword tin cậy -> luôn rơi vào fallback/unmapped
    "hotels/other lodgings": ["hotel", "motel", "inn", "lodging", "bed and breakfast", "hostel"],
    "dance clubs": ["night club", "dance club", "disco"],
    "swimming pools": ["swimming pool", "water park", "public pool"],
    "gyms": ["gym", "fitness center", "health club"],
    "bakeries": ["bakery", "pastry shop"],
    "beauty & spas": ["spa", "salon", "beauty", "nail salon", "massage"],
    "cafes": ["cafe", "coffee shop", "coffee"],
    "view points": ["scenic", "lookout", "viewpoint", "observation deck"],
    "monuments": ["monument", "memorial", "historical landmark"],
    "gardens": ["garden", "botanical", "arboretum"],
    "restaurants": ["restaurant", "diner", "eatery"],  # đặt cuối vì "restaurant" là từ tổng quát nhất
}


# ---------------------------------------------------------------------------
# 2. Load Google Local metadata -> count category frequency
# ---------------------------------------------------------------------------
def load_google_categories(path: Path) -> Counter:
    """Đọc file meta-<State>.json.gz (1 JSON object / dòng), đếm tần suất
    xuất hiện của từng category text (mỗi business có thể có nhiều category)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}.\n"
            f"Hãy tải thủ công từ https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/ "
            f"rồi đặt đường dẫn đúng vào META_PATH ở đầu file."
        )
    counter = Counter()
    n_lines, n_skipped = 0, 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            try:
                biz = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue
            for cat in biz.get("category") or []:
                if cat:
                    counter[cat.strip()] += 1
    print(f"[load] {sum(counter.values())} lượt category trên {len(counter)} category text duy nhất")
    if n_skipped:
        print(f"[load] CẢNH BÁO: bỏ qua {n_skipped}/{n_lines} dòng ({n_skipped/n_lines:.2%}) "
              f"do lỗi parse JSON — kiểm tra lại file nếu tỷ lệ này cao bất thường")
    return counter


# ---------------------------------------------------------------------------
# 3. Keyword matching (tầng 1)
# ---------------------------------------------------------------------------
def keyword_match(google_category: str) -> str | None:
    """So khớp bằng regex word-boundary, KHÔNG dùng substring thô — tránh
    false positive kiểu 'bar' khớp nhầm bên trong 'Barbershop'/'Barbecue',
    hoặc 'mall' khớp nhầm bên trong 'Small business services'."""
    text = normalize_text(google_category)
    for uci_cat, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            kw_norm = normalize_text(kw)
            if re.search(rf"\b{re.escape(kw_norm)}\b", text):
                return uci_cat
    return None


# ---------------------------------------------------------------------------
# 4. Embedding fallback (tầng 2) — chỉ chạy cho phần keyword không bắt được
# ---------------------------------------------------------------------------
def embedding_match(unmatched_categories: list[str]) -> dict[str, tuple[str, float]]:
    from sentence_transformers import SentenceTransformer, util

    print(f"[embedding] Đang tính similarity cho {len(unmatched_categories)} category chưa khớp keyword...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    uci_emb = model.encode(UCI_CATEGORIES, convert_to_tensor=True)
    goog_emb = model.encode(unmatched_categories, convert_to_tensor=True)

    sims = util.cos_sim(goog_emb, uci_emb)  # [n_unmatched, 24]
    results = {}
    for i, cat in enumerate(unmatched_categories):
        best_idx = int(sims[i].argmax())
        best_score = float(sims[i][best_idx])
        results[cat] = (UCI_CATEGORIES[best_idx], best_score)
    return results


# ---------------------------------------------------------------------------
# 5. Build full crosswalk table
# ---------------------------------------------------------------------------
def build_crosswalk(category_freq: Counter) -> pd.DataFrame:
    rows = []
    unmatched = []

    for cat, freq in category_freq.items():
        uci_match = keyword_match(cat)
        if uci_match:
            rows.append({"google_category": cat, "uci_category": uci_match,
                          "method": "keyword", "similarity": 1.0, "frequency": freq})
        else:
            unmatched.append(cat)

    if unmatched:
        emb_results = embedding_match(unmatched)
        for cat in unmatched:
            uci_match, score = emb_results[cat]
            freq = category_freq[cat]
            if score >= EMBEDDING_SIM_THRESHOLD:
                rows.append({"google_category": cat, "uci_category": uci_match,
                             "method": "embedding", "similarity": round(score, 3), "frequency": freq})
            else:
                rows.append({"google_category": cat, "uci_category": "unmapped",
                             "method": "unmapped", "similarity": round(score, 3), "frequency": freq})

    df = pd.DataFrame(rows).sort_values(["uci_category", "frequency"], ascending=[True, False])
    return df.reset_index(drop=True)


def print_coverage_report(df: pd.DataFrame):
    total_freq = df["frequency"].sum()
    mapped_freq = df.loc[df["uci_category"] != "unmapped", "frequency"].sum()
    print("\n--- Coverage report ---")
    print(f"Tổng lượt category: {total_freq}")
    print(f"Đã map vào 1 trong 24 UCI category: {mapped_freq} ({mapped_freq/total_freq:.1%})")
    print(f"Chưa map được (unmapped): {total_freq - mapped_freq} ({1 - mapped_freq/total_freq:.1%})")

    print("\nSố lượng google category text map vào từng UCI category (theo method):")
    print(df.groupby(["uci_category", "method"]).size().unstack(fill_value=0))

    empty_uci = set(UCI_CATEGORIES) - set(df.loc[df["uci_category"] != "unmapped", "uci_category"])
    if empty_uci:
        print(f"\n[CẢNH BÁO] Các UCI category KHÔNG map được từ google category nào trong dữ liệu này: "
              f"{sorted(empty_uci)}")
        print("-> Đúng như đã lường trước: 'local services' và các category hẹp khác "
              "thường không có tương đương rõ ràng. Cần quyết định: bỏ qua category này "
              "khi tính hybrid score, hoặc coi nó là NaN (không rated) cho mọi business Google Local.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    category_freq = load_google_categories(META_PATH)
    crosswalk_df = build_crosswalk(category_freq)
    print_coverage_report(crosswalk_df)

    out_path = OUTPUT_DIR / "category_crosswalk.csv"
    crosswalk_df.to_csv(out_path, index=False)
    print(f"\n[save] Crosswalk table -> {out_path}")
    print("Review file này thủ công trước khi dùng: đặc biệt các dòng method='embedding' "
          "(suy đoán từ độ tương đồng ngữ nghĩa, có thể sai) và mọi dòng uci_category='unmapped'.")

    return crosswalk_df


if __name__ == "__main__":
    main()
